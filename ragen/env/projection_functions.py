"""
Action Projection Functions for converting LLM text output to environment actions.
These functions handle the translation between natural language responses and
discrete environment actions.
"""

import re
from typing import List, Tuple, Callable


def create_projection_function(env_type: str) -> Callable:
    """
    Factory function to create environment-specific projection functions.
    
    Args:
        env_type: Type of environment (e.g., 'sokoban', 'webshop', 'countdown')
        
    Returns:
        Projection function that converts text to actions
    """
    if env_type == 'sokoban':
        return sokoban_projection
    elif env_type == 'webshop':
        return webshop_projection
    elif env_type == 'countdown':
        return countdown_projection
    elif env_type == 'metamathqa':
        return metamathqa_projection
    elif env_type == 'frozen_lake':
        return frozen_lake_projection
    elif env_type == 'bandit':
        return bandit_projection
    else:
        return generic_projection


def sokoban_projection(text_actions: List[str], 
                      action_spaces: List[List] = None) -> Tuple[List[int], List[bool]]:
    """
    Convert LLM text output to Sokoban actions.
    
    Args:
        text_actions: Raw text from language model
        action_spaces: Available actions for each environment (unused for Sokoban)
        
    Returns:
        processed_actions: Integer action IDs (1=Up, 2=Down, 3=Left, 4=Right)
        validity_flags: Whether each action is valid
    """
    processed_actions = []
    validity_flags = []
    
    # Sokoban action mapping (matches SokobanEnvConfig exactly!)
    # From ragen/env/sokoban/config.py: {1:"Up", 2:"Down", 3:"Left", 4:"Right"}
    action_mapping = {
        'up': 1,
        'down': 2, 
        'left': 3,
        'right': 4,
        # Add synonyms
        'north': 1,
        'south': 2,
        'west': 3,
        'east': 4,
        # Add single letter shortcuts
        'u': 1,
        'd': 2,
        'l': 3,
        'r': 4,
    }
    
    for text_action in text_actions:
        # First try to extract action from XML tags
        action_match = re.search(r'<action>(.*?)</action>', text_action.lower())
        
        if action_match:
            clean_action = action_match.group(1).strip().lower()
        else:
            # If no XML tags, treat the entire text as the action (like single-process version)
            clean_action = text_action.strip().lower()
        
        # Map text to action ID
        if clean_action in action_mapping:
            processed_actions.append(action_mapping[clean_action])
            validity_flags.append(True)
        else:
            # Try partial matching for complex phrases
            found = False
            for key, value in action_mapping.items():
                if key in clean_action:
                    processed_actions.append(value)
                    validity_flags.append(True)
                    found = True
                    break
            
            if not found:
                processed_actions.append(1)  # Default to 'up'
                validity_flags.append(False)
    
    return processed_actions, validity_flags


def webshop_projection(text_actions: List[str], 
                      action_spaces: List[List] = None) -> Tuple[List[str], List[bool]]:
    """
    Convert LLM text output to WebShop actions.
    
    Args:
        text_actions: Raw text from language model
        action_spaces: Available actions for each environment
        
    Returns:
        processed_actions: Clean action strings
        validity_flags: Whether each action is valid
    """
    processed_actions = []
    validity_flags = []
    
    for i, text_action in enumerate(text_actions):
        # Extract action from XML tags
        action_match = re.search(r'<action>(.*?)</action>', text_action)
        
        if action_match:
            clean_action = action_match.group(1).strip()
            
            # Validate action format for WebShop
            is_valid = True
            
            # Check for common WebShop action patterns
            if not (clean_action.startswith('search[') or 
                   clean_action.startswith('click[') or
                   clean_action == 'buy now' or
                   clean_action == 'back'):
                is_valid = False
            
            processed_actions.append(clean_action)
            validity_flags.append(is_valid)
        else:
            # Try to extract action patterns directly from text
            # Look for search[...] or click[...] patterns
            search_match = re.search(r'search\[(.*?)\]', text_action)
            click_match = re.search(r'click\[(.*?)\]', text_action)
            
            if search_match:
                clean_action = f"search[{search_match.group(1)}]"
                processed_actions.append(clean_action)
                validity_flags.append(True)
            elif click_match:
                clean_action = f"click[{click_match.group(1)}]"
                processed_actions.append(clean_action)
                validity_flags.append(True)
            elif 'buy' in text_action.lower():
                processed_actions.append("buy now")
                validity_flags.append(True)
            else:
                # Fallback for malformed actions
                processed_actions.append("click[back to search]")  # Safe default action
                validity_flags.append(False)
    
    return processed_actions, validity_flags


def countdown_projection(text_actions: List[str], 
                        action_spaces: List[List] = None) -> Tuple[List[str], List[bool]]:
    """
    Convert LLM text output to Countdown math actions.
    
    Args:
        text_actions: Raw text from language model
        action_spaces: Available actions for each environment
        
    Returns:
        processed_actions: Mathematical expressions
        validity_flags: Whether each action is valid
    """
    processed_actions = []
    validity_flags = []
    
    for text_action in text_actions:
        # Try to extract mathematical expression directly (no XML tags needed)
        # Look for patterns like "- 1 + 2 + 3" or "1 + 2 - 3"
        math_patterns = [
            r'([+\-]?\s*\d+(?:\s*[+\-*/]\s*\d+)*)',  # Mathematical expression
            r'(\d+\s*[+\-*/]\s*\d+(?:\s*[+\-*/]\s*\d+)*)',  # Standard math expression
        ]
        
        clean_action = None
        for pattern in math_patterns:
            math_match = re.search(pattern, text_action)
            if math_match:
                clean_action = math_match.group(1).strip()
                break
        
        if clean_action is None:
            # Fallback: use the entire text as action
            clean_action = text_action.strip()
        
        # Validate mathematical expression
        is_valid = _is_valid_math_expression(clean_action)
        
        processed_actions.append(clean_action)
        validity_flags.append(is_valid)
    
    return processed_actions, validity_flags


def metamathqa_projection(text_actions: List[str], 
                         action_spaces: List[List] = None) -> Tuple[List[str], List[bool]]:
    """
    Convert LLM text output to MetaMathQA actions.
    
    Args:
        text_actions: Raw text from language model
        action_spaces: Available actions for each environment
        
    Returns:
        processed_actions: Answer strings (entire response text)
        validity_flags: Whether each action is valid
    """
    processed_actions = []
    validity_flags = []
    
    for text_action in text_actions:
        # For MetaMathQA, we pass the entire response text as the action
        # The environment will handle answer extraction internally
        clean_action = text_action.strip()
        
        # Consider valid if it's non-empty
        is_valid = len(clean_action) > 0
        
        processed_actions.append(clean_action)
        validity_flags.append(is_valid)
    
    return processed_actions, validity_flags


def frozen_lake_projection(text_actions: List[str], 
                          action_spaces: List[List] = None) -> Tuple[List[int], List[bool]]:
    """
    Convert LLM text output to FrozenLake actions.
    
    Args:
        text_actions: Raw text from language model
        action_spaces: Available actions for each environment
        
    Returns:
        processed_actions: Integer action IDs (1=Left, 2=Down, 3=Right, 4=Up)
        validity_flags: Whether each action is valid
    """
    processed_actions = []
    validity_flags = []
    
    # FrozenLake action mapping (matches FrozenLakeEnvConfig)
    action_mapping = {
        'left': 1,
        'down': 2,
        'right': 3,
        'up': 4,
        'west': 1,
        'south': 2,
        'east': 3,
        'north': 4,
    }
    
    for text_action in text_actions:
        # Extract action from XML tags
        action_match = re.search(r'<action>(.*?)</action>', text_action.lower())
        
        if action_match:
            clean_action = action_match.group(1).strip().lower()
            
            # Map text to action ID
            if clean_action in action_mapping:
                processed_actions.append(action_mapping[clean_action])
                validity_flags.append(True)
            else:
                # Try partial matching
                found = False
                for key, value in action_mapping.items():
                    if key in clean_action:
                        processed_actions.append(value)
                        validity_flags.append(True)
                        found = True
                        break
                
                if not found:
                    processed_actions.append(1)  # Default to 'left'
                    validity_flags.append(False)
        else:
            # No action tags found
            processed_actions.append(1)  # Default to 'left'
            validity_flags.append(False)
    
    return processed_actions, validity_flags


def bandit_projection(text_actions: List[str], 
                     action_spaces: List[List] = None) -> Tuple[List[str], List[bool]]:
    """
    Convert LLM text output to Bandit actions.
    
    Args:
        text_actions: Raw text from language model
        action_spaces: Available actions for each environment
        
    Returns:
        processed_actions: Arm names (phoenix/dragon)
        validity_flags: Whether each action is valid
    """
    processed_actions = []
    validity_flags = []
    
    for text_action in text_actions:
        # Extract action from answer tags (MetaMathQA style)
        action_match = re.search(r'<answer>(.*?)</answer>', text_action.lower())
        
        if action_match:
            clean_action = action_match.group(1).strip().lower()
            
            # Check for valid arm names
            if 'phoenix' in clean_action:
                processed_actions.append('phoenix')
                validity_flags.append(True)
            elif 'dragon' in clean_action:
                processed_actions.append('dragon')
                validity_flags.append(True)
            else:
                processed_actions.append('phoenix')  # Default to phoenix
                validity_flags.append(False)
        else:
            # Try to find arm names directly in text
            text_lower = text_action.lower()
            if 'dragon' in text_lower:
                processed_actions.append('dragon')
                validity_flags.append(True)
            elif 'phoenix' in text_lower:
                processed_actions.append('phoenix')
                validity_flags.append(True)
            else:
                processed_actions.append('phoenix')  # Default to phoenix
                validity_flags.append(False)
    
    return processed_actions, validity_flags


def generic_projection(text_actions: List[str], 
                      action_spaces: List[List] = None) -> Tuple[List[str], List[bool]]:
    """
    Generic projection function for unknown environment types.
    
    Args:
        text_actions: Raw text from language model
        action_spaces: Available actions for each environment
        
    Returns:
        processed_actions: Clean action strings
        validity_flags: Whether each action is valid
    """
    processed_actions = []
    validity_flags = []
    
    for text_action in text_actions:
        # Extract action from XML tags
        action_match = re.search(r'<action>(.*?)</action>', text_action)
        
        if action_match:
            clean_action = action_match.group(1).strip()
            processed_actions.append(clean_action)
            validity_flags.append(True)
        else:
            # Use the entire text as action
            processed_actions.append(text_action.strip())
            validity_flags.append(False)  # Mark as invalid since no proper format
    
    return processed_actions, validity_flags


def _is_valid_math_expression(expr: str) -> bool:
    """
    Validate if a string is a valid mathematical expression.
    
    Args:
        expr: Expression to validate
        
    Returns:
        True if valid mathematical expression
    """
    try:
        # Simple validation - check for basic math operations
        if re.match(r'^[\d\s+\-*/().]+$', expr):
            # Try to evaluate (safely)
            eval(expr)
            return True
        return False
    except:
        return False 