# Neo4j connection parameters
uri = "bolt://localhost:7687"
username = "neo4j"
password = "yzx010203"

import pandas as pd
from neo4j import GraphDatabase

# Create Neo4j driver
driver = GraphDatabase.driver(uri, auth=(username, password))

# Define a function to add properties to existing nodes
def add_properties_to_node(tx, entity_name, property_name, property_value):
    # Update the properties of an existing node
    query = """
    MATCH (n {name: $name})
    SET n[$property_name] = $property_value
    RETURN n
    """
    # Execute the query and print the information about the added property
    result = tx.run(query, name=entity_name, property_name=property_name, property_value=property_value)

    print(f"Added property '{property_name}' with value '{property_value}' to node '{entity_name}'")

# Read the Excel file and process each row of data
excel_file = r'graph_data\attribute.xlsx'
with driver.session() as session:
    df = pd.read_excel(excel_file)
    # Iterate through each row in the Excel file
    for index, row in df.iterrows():
        entity_name = row.iloc[0]
        property_name = row.iloc[1]
        property_value = row.iloc[2]
        # Use execute_write to update the database and output the log during each update
        session.execute_write(add_properties_to_node, entity_name, property_name, property_value)

print("All properties and values have been successfully added to the nodes.")

# Close the Neo4j driver
driver.close()
