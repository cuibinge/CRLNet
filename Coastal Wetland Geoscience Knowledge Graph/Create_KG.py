from py2neo import Graph, Node, Relationship
import pandas as pd

# Initialize connection to the Neo4j database
g = Graph('bolt://localhost:7687', user='neo4j', password='yzx010203')

# Define the function to create nodes and relationships
def create_nodes_and_relationship(start_label, start_name, end_label, end_name, relation_type):
    start_node = Node(start_label, name=start_name)
    end_node = Node(end_label, name=end_name)
    relation = Relationship(start_node, relation_type, end_node)
    g.merge(start_node, start_label, "name")
    g.merge(end_node, end_label, "name")
    g.merge(relation, start_label, "name")


file1 = r'graph_data\relation.xlsx'
file2 = r'graph_data\entity.xlsx'
df1 = pd.read_excel(file1, engine='openpyxl')
df2 = pd.read_excel(file2, engine='openpyxl')


tantu = df2['Tantu'].dropna().tolist()  # Get all values from 'Tantu' column
Community = df2['Community'].dropna().tolist()  # Get all values from 'Community' column
WaterBody = df2['WaterBody'].dropna().tolist()  # Get all values from 'WaterBody' column
WetLand = df2['WetLand'].dropna().tolist()  # Get all values from 'WetLand' column

# Iterate through each row in the Excel DataFrame
for index, row in df1.iterrows():
    print(f"Processing row {index + 1}: {row.to_dict()}")
    # Extract the values for each column
    entity1 = row.iloc[0]  # First entity
    relationship = row.iloc[1]  # Relationship type
    entity2 = row.iloc[2]  # Second entity
    # Based on the relationship, create nodes and relationships
    if 'Habitatize' in relationship:
        if entity2 in tantu and entity1 in Community:
            create_nodes_and_relationship("Community", entity1, "Mudflat", entity2, relationship)
        elif entity2 in tantu:
            create_nodes_and_relationship("Plant", entity1, "Mudflat", entity2, relationship)
        elif entity2 in WaterBody:
            create_nodes_and_relationship("Community", entity1, "WaterBody", entity2, relationship)
        elif entity2 in WetLand:
            create_nodes_and_relationship("Community", entity1, "WetLand", entity2, relationship)
        else:
            create_nodes_and_relationship("Plant", entity1, "WaterBody", entity2, relationship)
    elif 'Common name' in relationship or 'Alias' in relationship:
        if entity1 in WaterBody:
            create_nodes_and_relationship("WaterBody", entity1, "WaterBody", entity2, relationship)
        elif entity1 == "Tidal flat":
            create_nodes_and_relationship("Mudflat", entity1, "Mudflat", entity2, relationship)
        else:
            create_nodes_and_relationship("Plant", entity1, "Plant", entity2, relationship)
    elif 'Dominate' in relationship or 'Symbiotize' in relationship:
        create_nodes_and_relationship("Community", entity1, "Plant", entity2, relationship)
    elif 'Lianthong' in relationship:
        create_nodes_and_relationship("Community", entity1, "Community", entity2, relationship)
    elif 'Kingdom' in relationship:
        create_nodes_and_relationship("Plant", entity1, "Kingdom", entity2, relationship)
    elif 'Phylum' in relationship:
        create_nodes_and_relationship("Plant", entity1, "Phylum", entity2, relationship)
    elif 'Class' in relationship:
        create_nodes_and_relationship("Plant", entity1, "Class", entity2, relationship)
    elif 'Order' in relationship:
        create_nodes_and_relationship("Plant", entity1, "Order", entity2, relationship)
    elif 'Family' in relationship:
        create_nodes_and_relationship("Plant", entity1, "Family", entity2, relationship)
    elif 'Genus' in relationship:
        create_nodes_and_relationship("Plant", entity1, "Genus", entity2, relationship)
    elif 'Include' in relationship:
        create_nodes_and_relationship("WetLand", entity1, "WetLand", entity2, relationship)
    elif 'Contain' in relationship:
        create_nodes_and_relationship("WaterBody", entity1, "Plant", entity2, relationship)
    elif 'next to' in relationship:
        if entity1 in WaterBody:
            if entity2 in WaterBody:
                create_nodes_and_relationship("WaterBody", entity1, "WaterBody", entity2, relationship)
            elif entity2 in tantu:
                create_nodes_and_relationship("WaterBody", entity1, "Mudflat", entity2, relationship)
            elif entity2 in Community:
                create_nodes_and_relationship("WaterBody", entity1, "Community", entity2, relationship)
            else:
                create_nodes_and_relationship("WaterBody", entity1, "Plant", entity2, relationship)
        elif entity1 in Community:
            if entity2 in Community:
                create_nodes_and_relationship("Community", entity1, "Community", entity2, relationship)
            elif entity2 in tantu:
                create_nodes_and_relationship("Community", entity1, "Mudflat", entity2, relationship)
            elif entity2 in WaterBody:
                create_nodes_and_relationship("Community", entity1, "WaterBody", entity2, relationship)
            else:
                create_nodes_and_relationship("Community", entity1, "Plant", entity2, relationship)
    elif 'far from' in relationship:
        if entity1 in WaterBody:
            if entity2 in WaterBody:
                create_nodes_and_relationship("WaterBody", entity1, "WaterBody", entity2, relationship)
            elif entity2 in Community:
                create_nodes_and_relationship("WaterBody", entity1, "Community", entity2, relationship)
            elif entity2 in tantu:
                create_nodes_and_relationship("WaterBody", entity1, "Mudflat", entity2, relationship)
            else:
                create_nodes_and_relationship("WaterBody", entity1, "Plant", entity2, relationship)
        elif entity1 in Community:
            if entity2 in tantu:
                create_nodes_and_relationship("Community", entity1, "Mudflat", entity2, relationship)
            elif entity2 in Community:
                create_nodes_and_relationship("Community", entity1, "Community", entity2, relationship)
            else:
                create_nodes_and_relationship("Community", entity1, "Plant", entity2, relationship)
        else:
            if entity2 in tantu:
                create_nodes_and_relationship("Plant", entity1, "Mudflat", entity2, relationship)
            else:
                create_nodes_and_relationship("Plant", entity1, "Plant", entity2, relationship)
    elif 'Overlap' in relationship or 'depend on' in relationship or 'Buffer' in relationship:
        create_nodes_and_relationship("Community", entity1, "WaterBody", entity2, relationship)
    elif 'Cover' in relationship or 'connect to' in relationship:
        create_nodes_and_relationship("WaterBody", entity1, "Plant", entity2, relationship)
    else:
        continue
