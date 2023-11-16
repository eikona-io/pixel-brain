
# Pixelbrain

Pixelbrain is a project that uses machine learning models to easily and automatically process and classify images.  

It includes modules for image Q&A with GPT-4 Vision, image clustering using embedding models and vector search, image classification with models such as ResNet, preprocessing modules for different models, and a database for storing and retrieving processed data.  
All the modules are composable and extendable.

The project also includes pre-built apps for purposes such as people identification.

## Installation  
To install Pixel Brain, you can use pip to install directly from the GitHub repository. Run the following command:

```bash
pip install git+https://github.com/omerhac/pixel-brain.git
```

## Usage
```bash
# pre-built identity-tagger application
tag_identity --data_path /path/to/your/data --export /path/to/export.scv

tag_identity -h # for more options
```



## High level design
![High Level Design](assets/hld.png)

## Modules

### Preprocessor

This is an interface for preprocessing a batch of images for a certain model. It is an abstract base class and needs to be subclassed for specific preprocessing methods.

### DataLoader

The DataLoader class loads and decodes images either from disk or S3. It can be configured to load images in batches and optionally decode the images.

### Database

The Database class is used to interact with the MongoDB database. It can store fields, query vector fields, find images, and perform other database operations.

### Gpt4VModule

This module processes images with GPT-4 Vision and stores the results in a database. It can ask a question to GPT-4 Vision and store the results in a specified field in the database.

### ResnetClassifierModule

This module classifies images into one of the ImageNet classes and stores the class in a database. It can receive a list of classes to choose from (a subset of ImageNet classes), out of which it will pick the one with the largest probability.

### FacenetEmbedderModule

This module is used to embed images using the FaceNet model. It crops out faces from the images and then embed's them in a vector database (ChromaDB)

### PeopleIdentifierModule

This module is used to identify people in images. The module processes the images and assigns identities to them based on the embeddings stored in the database.