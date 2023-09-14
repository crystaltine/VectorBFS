def _get_progress_string(progress: float, length: int = 20) -> str:
    big_block = "█"
    empty_block = "\x1B[2m█\033[0m"
    
    num_blocks = int(round(progress * length))
    num_empty = length - num_blocks
    
    if num_blocks == length:
        return "[" + big_block * num_blocks + "]"
    
    return "[" + big_block * num_blocks + empty_block * num_empty + "]"