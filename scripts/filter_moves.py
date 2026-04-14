import json
import os

def filter_moves(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        try:
            moves = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    filtered_moves = {}    
    skip_nonstandard = ["Past", "LGPE", "CAP", "Gigantamax", "Unobtainable"]
    
    for move_id, move_data in moves.items():
        # Skip if explicitly marked as non-standard for Gen 9
        nonstandard = move_data.get("isNonstandard")
        if nonstandard in skip_nonstandard:
            continue
            
        # Skip if it is a pure Z-Move or Max Move
        if "isZ" in move_data or "isMax" in move_data:
            continue
            
        new_move_data = move_data.copy()
        
        # Remove redundant fields
        # name already encoded and pp part of numerical embedding
        redundant_fields = ["num", "name", "pp", "desc", "contestType", "zMove", "maxMove"]
        for field in redundant_fields:
            if field in new_move_data:
                del new_move_data[field]
        
        filtered_moves[move_id] = new_move_data

    with open(output_file, 'w') as f:
        json.dump(filtered_moves, f, indent=2)
    
    print(f"Filtered {len(moves)} moves down to {len(filtered_moves)} moves.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Assuming the script is run from the project root
    input_path = "data/moves.json"
    output_path = "data/filtered_moves.json"
    filter_moves(input_path, output_path)
