def update_policy(self, data: DataProto):
    # Original implementation setup
    self.actor_module.train()
    assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
    temperature = data.meta_info['temperature']

    # Process first batch only for comparison
    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
    if self.config.state_masking:
        select_keys.append('loss_mask')
    if self.config.use_kl_loss:
        select_keys.append('ref_log_prob')
    batch = data.select(batch_keys=select_keys).batch
    dataloader = batch.split(self.config.ppo_mini_batch_size)
    
    # Get first batch for testing
    first_batch = next(iter(dataloader))
    if self.config.use_dynamic_bsz:
        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
        micro_batches, _ = rearrange_micro_batches(batch=first_batch, max_token_len=max_token_len)
    else:
        micro_batches = first_batch.split(self.config.ppo_micro_batch_size)

    # Save initial parameters
    pre_update_params = {name: param.data.clone() for name, param in self.actor_module.named_parameters()}
    
    # 1. Run original approach (no merging)
    self.actor_optimizer.zero_grad()
    original_metrics = {}
    
    for data in micro_batches:
        # Original code for processing micro_batch and computing loss
        data = data.cuda()
        # ... [omitting all the processing code for brevity] ...
        policy_loss = pg_loss - entropy_loss * entropy_coeff
        if self.config.use_kl_loss:
            # ... KL loss computation ...
            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
        
        # Original backward call
        loss = policy_loss / self.gradient_accumulation
        loss.backward()
    
    # Capture original gradients before optimizer step
    original_grads = {name: param.grad.clone() if param.grad is not None else None 
                      for name, param in self.actor_module.named_parameters()}
    
    # Reset optimizer to start fresh
    self.actor_optimizer.zero_grad()
    
    # 2. Now run optimized approach (with merging)
    is_peft_model = isinstance(self.actor_module._fsdp_wrapped_module, PeftModel)
    
    # Merge adapter if using PEFT
    if is_peft_model:
        with FSDP.summon_full_params(self.actor_module):
            self.actor_module.merge_adapter()
            
    all_policy_losses = []
    
    for data in micro_batches:
        # Same processing as before, but store losses instead of immediate backward
        data = data.cuda()
        # ... [processing code] ...
        policy_loss = pg_loss - entropy_loss * entropy_coeff
        if self.config.use_kl_loss:
            # ... KL loss computation ...
            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
            
        all_policy_losses.append(policy_loss)
    
    # Unmerge adapter before backward
    if is_peft_model:
        with FSDP.summon_full_params(self.actor_module):
            self.actor_module.unmerge_adapter()
    
    # Do backward on total loss
    total_loss = torch.sum(torch.stack(all_policy_losses)) / self.gradient_accumulation
    total_loss.backward()
    
    # Capture optimized approach gradients
    optimized_grads = {name: param.grad.clone() if param.grad is not None else None 
                       for name, param in self.actor_module.named_parameters()}
    
    # Compare gradients
    max_grad_diff = 0
    for name in original_grads:
        if original_grads[name] is not None and optimized_grads[name] is not None:
            diff = torch.max(torch.abs(original_grads[name] - optimized_grads[name]))
            max_grad_diff = max(max_grad_diff, diff.item())
    
    print(f"Maximum gradient difference: {max_grad_diff}")
    assert max_grad_diff < 1e-5, "Gradient mismatch detected"
    
    # Now continue with your normal training process
    # ... [rest of original update_policy] ...