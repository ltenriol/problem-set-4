'''
PART 2: NETWORK CENTRALITY METRICS

Using the imbd_movies dataset
- Build a graph and perform some rudimentary graph analysis, extracting centrality metrics from it. 
- Below is some basic code scaffolding that you will need to add to
- Tailor this code scaffolding and its stucture to however works to answer the problem
- Make sure the code is inline with the standards we're using in this class 
'''

import numpy as np
import pandas as pd
import networkx as nx
import json

# Build the graph
g = nx.Graph()

# Set up your dataframe(s) -> the df that's output to a CSV should include at least the columns 'left_actor_name', '<->', 'right_actor_name'


with open('data/imdb_movies_data.jsonl', 'r') as in_file:
    # Don't forget to comment your code
    for line in in_file:
        # Don't forget to include docstrings for all functions

        # Load the movie from this line
        this_movie = json.loads(line)
            
        # Create a node for every actor
        for actor_id,actor_name in this_movie['actors']:
        # add the actor to the graph    
            g.add_node(actor_id, name=actor_name)
        # Iterate through the list of actors, generating all pairs
        ## Starting with the first actor in the list, generate pairs with all subsequent actors
        ## then continue to second actor in the list and repeat
        
        i = 0 #counter
        for left_actor_id,left_actor_name in this_movie['actors']:
            for right_actor_id,right_actor_name in this_movie['actors'][i+1:]:
                if g.has_edge(left_actor_id, right_actor_id):
                    g[left_actor_id][right_actor_id]["weight"] += 1
                else:
                    g.add_edge(left_actor_id, right_actor_id, weight=1)

                # Get the current weight, if it exists
                current_weight = g[left_actor_id][right_actor_id]["weight"] if g.has_edge(left_actor_id, right_actor_id) else 0
                
                # Add an edge for these actors
                g.add_edge(left_actor_id, right_actor_id, weight=current_weight + 1)
                
                


# Print the info below
print("Nodes:", len(g.nodes))

#Print the 10 the most central nodes
centrality = nx.degree_centrality(g)
centrality_df = pd.DataFrame(centrality.items(), columns=['actor_id', 'centrality'])
centrality_df = centrality_df.sort_values(by='centrality', ascending=False).head(10)
print("Top 10 most central nodes:")
print(centrality_df)


# Output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`
output_path = 'data/network_centrality.csv'
centrality_df.to_csv(output_path, index=False)  