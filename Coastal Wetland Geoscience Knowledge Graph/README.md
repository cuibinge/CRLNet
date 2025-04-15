# knowledge_graph
**This project is designed to showcase the coastal wetland geological knowledge graph constructed from 14k collected relationship and attribute triples.**

### Before running the code, establish a connection to the Neo4j database.
````python
 g = Graph('bolt://localhost:7687', user='your user', password='your password')
````
##  *Load relationships from the Excel file*
### 1. The Excel file is stored in the 'graph_data' folder, and you can add or remove files as needed.
### 2. Run the 'Create_KG.py' script.
You can replace the file with the name of your relationship file,
````python
 file1 = r'graph_data\relation.xlsx'
````
and add the corresponding entities in this file to meet your specific requirements.
````python
 file2 = r'graph_data\entity.xlsx'
````

## *Process for adding properties to nodes*
### 1. Ensure that the node has already been created before adding properties.
### 2. Add the property triples to the 'graph_data\attribute.xlsx' file.
You can replace this file with the name of your attribute file as needed.
````python
 excel_file = r'graph_data\attribute.xlsx'
````
### 3. Run the 'Add_attribute.py' script.
