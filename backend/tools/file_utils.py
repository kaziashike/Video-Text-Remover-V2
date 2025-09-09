import os
import logging

class Filesplit:
    """
    Custom implementation to replace the problematic fsplit package.
    Used for merging split files in the project.
    """
    
    def merge(self, input_dir):
        """
        Merge split files in the input directory.
        Looks for files with the pattern 'filename.ext.split.[number]' and merges them.
        """
        if not os.path.exists(input_dir):
            logging.warning(f"Directory {input_dir} does not exist")
            return
            
        # Get all files in the directory
        files = os.listdir(input_dir)
        
        # Group files by their base name
        file_groups = {}
        for file in files:
            # Check if it's a split file (has .split. in the name)
            if '.split.' in file:
                # Extract base name (everything before .split.)
                base_name = '.split.'.join(file.split('.split.')[:-1])
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(file)
        
        # Sort each group by split number and merge
        for base_name, split_files in file_groups.items():
            # Sort files by their split number
            split_files.sort(key=lambda x: int(x.split('.split.')[-1]))
            
            # Merge files
            output_path = os.path.join(input_dir, base_name)
            try:
                with open(output_path, 'wb') as output_file:
                    for split_file in split_files:
                        split_path = os.path.join(input_dir, split_file)
                        with open(split_path, 'rb') as f:
                            output_file.write(f.read())
                
                logging.info(f"Merged {len(split_files)} parts into {output_path}")
            except Exception as e:
                logging.error(f"Error merging files for {base_name}: {e}")