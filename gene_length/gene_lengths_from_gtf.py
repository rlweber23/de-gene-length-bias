import pandas as pd

def generate_gene_length_table(gtf_filepath, output_file = None):

    gene_lengths = []

    # Read through the GTF file
    with open(gtf_filepath, "r") as f:
    
        for line in f:
    
            # Skip comment lines
            if line.startswith("#"):
                continue
    
            # Split line into columns
            parts = line.strip().split("\t")
    
            # Only use genes
            if parts[2] != "gene":
                continue
    
            # Get gene start and end positions
            start = int(parts[3])
            end = int(parts[4])
    
            # Calculate gene length
            gene_length = end - start + 1
    
            # Get gene ID
            attributes = parts[8]
            gene_id = attributes.split('gene_id "')[1].split('"')[0]
    
            # Save gene ID and length
            gene_lengths.append({
                "geneID": gene_id,
                "gene_length": gene_length
            })
    
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(gene_lengths)


    if output_file:
        df.to_csv(output_file, index=False)

        print("Saved gene lengths to:", output_file)


    print("Finished!")
    
    return(df)


