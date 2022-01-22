{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Breast Cancer_Detection_K-Fold.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOop9qyCrvDW0esKGDAPScZ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/prai96/DEEP_LEARNING/blob/main/Breast_Cancer_Detection_K_Fold.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgZG8gewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwoKICAgICAgbGV0IHBlcmNlbnREb25lID0gZmlsZURhdGEuYnl0ZUxlbmd0aCA9PT0gMCA/CiAgICAgICAgICAxMDAgOgogICAgICAgICAgTWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCk7CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPSBgJHtwZXJjZW50RG9uZX0lIGRvbmVgOwoKICAgIH0gd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCk7CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 90
        },
        "id": "8Ty0FKJhN_Ys",
        "outputId": "a7ef5ab0-cc17-4607-d1e9-3fd156892dbd"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-49f79258-7abe-4c63-8ff5-c3617ab2cc18\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-49f79258-7abe-4c63-8ff5-c3617ab2cc18\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving data.csv to data.csv\n",
            "User uploaded file \"data.csv\" with length 125204 bytes\n"
          ]
        }
      ],
      "source": [
        "#importing data\n",
        "from google.colab import files\n",
        "\n",
        "uploaded = files.upload()\n",
        "\n",
        "for fn in uploaded.keys():\n",
        "  print('User uploaded file \"{name}\" with length {length} bytes'.format(\n",
        "      name=fn, length=len(uploaded[fn])))\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#importing libraries\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import io\n",
        "import keras\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense\n",
        "from keras.wrappers.scikit_learn import KerasClassifier\n",
        "from sklearn.model_selection import cross_val_score\n",
        "import matplotlib.pyplot as plt\n"
      ],
      "metadata": {
        "id": "AjnRBaoqOoMF"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# data preprocessing\n",
        "data=pd.read_csv(io.StringIO(uploaded['data.csv'].decode('utf-8')))\n",
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 288
        },
        "id": "8laoIsiVQCGE",
        "outputId": "06f54210-b7a6-42ee-b372-a4bfefbec224"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "\n",
              "  <div id=\"df-ffd9b320-1e56-4663-9f07-d6837a008f46\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>diagnosis</th>\n",
              "      <th>radius_mean</th>\n",
              "      <th>texture_mean</th>\n",
              "      <th>perimeter_mean</th>\n",
              "      <th>area_mean</th>\n",
              "      <th>smoothness_mean</th>\n",
              "      <th>compactness_mean</th>\n",
              "      <th>concavity_mean</th>\n",
              "      <th>concave points_mean</th>\n",
              "      <th>symmetry_mean</th>\n",
              "      <th>fractal_dimension_mean</th>\n",
              "      <th>radius_se</th>\n",
              "      <th>texture_se</th>\n",
              "      <th>perimeter_se</th>\n",
              "      <th>area_se</th>\n",
              "      <th>smoothness_se</th>\n",
              "      <th>compactness_se</th>\n",
              "      <th>concavity_se</th>\n",
              "      <th>concave points_se</th>\n",
              "      <th>symmetry_se</th>\n",
              "      <th>fractal_dimension_se</th>\n",
              "      <th>radius_worst</th>\n",
              "      <th>texture_worst</th>\n",
              "      <th>perimeter_worst</th>\n",
              "      <th>area_worst</th>\n",
              "      <th>smoothness_worst</th>\n",
              "      <th>compactness_worst</th>\n",
              "      <th>concavity_worst</th>\n",
              "      <th>concave points_worst</th>\n",
              "      <th>symmetry_worst</th>\n",
              "      <th>fractal_dimension_worst</th>\n",
              "      <th>Unnamed: 32</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>842302</td>\n",
              "      <td>M</td>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>0.2419</td>\n",
              "      <td>0.07871</td>\n",
              "      <td>1.0950</td>\n",
              "      <td>0.9053</td>\n",
              "      <td>8.589</td>\n",
              "      <td>153.40</td>\n",
              "      <td>0.006399</td>\n",
              "      <td>0.04904</td>\n",
              "      <td>0.05373</td>\n",
              "      <td>0.01587</td>\n",
              "      <td>0.03003</td>\n",
              "      <td>0.006193</td>\n",
              "      <td>25.38</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>842517</td>\n",
              "      <td>M</td>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>0.1812</td>\n",
              "      <td>0.05667</td>\n",
              "      <td>0.5435</td>\n",
              "      <td>0.7339</td>\n",
              "      <td>3.398</td>\n",
              "      <td>74.08</td>\n",
              "      <td>0.005225</td>\n",
              "      <td>0.01308</td>\n",
              "      <td>0.01860</td>\n",
              "      <td>0.01340</td>\n",
              "      <td>0.01389</td>\n",
              "      <td>0.003532</td>\n",
              "      <td>24.99</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>84300903</td>\n",
              "      <td>M</td>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>0.2069</td>\n",
              "      <td>0.05999</td>\n",
              "      <td>0.7456</td>\n",
              "      <td>0.7869</td>\n",
              "      <td>4.585</td>\n",
              "      <td>94.03</td>\n",
              "      <td>0.006150</td>\n",
              "      <td>0.04006</td>\n",
              "      <td>0.03832</td>\n",
              "      <td>0.02058</td>\n",
              "      <td>0.02250</td>\n",
              "      <td>0.004571</td>\n",
              "      <td>23.57</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>84348301</td>\n",
              "      <td>M</td>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>0.2597</td>\n",
              "      <td>0.09744</td>\n",
              "      <td>0.4956</td>\n",
              "      <td>1.1560</td>\n",
              "      <td>3.445</td>\n",
              "      <td>27.23</td>\n",
              "      <td>0.009110</td>\n",
              "      <td>0.07458</td>\n",
              "      <td>0.05661</td>\n",
              "      <td>0.01867</td>\n",
              "      <td>0.05963</td>\n",
              "      <td>0.009208</td>\n",
              "      <td>14.91</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>84358402</td>\n",
              "      <td>M</td>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>0.1809</td>\n",
              "      <td>0.05883</td>\n",
              "      <td>0.7572</td>\n",
              "      <td>0.7813</td>\n",
              "      <td>5.438</td>\n",
              "      <td>94.44</td>\n",
              "      <td>0.011490</td>\n",
              "      <td>0.02461</td>\n",
              "      <td>0.05688</td>\n",
              "      <td>0.01885</td>\n",
              "      <td>0.01756</td>\n",
              "      <td>0.005115</td>\n",
              "      <td>22.54</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ffd9b320-1e56-4663-9f07-d6837a008f46')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-ffd9b320-1e56-4663-9f07-d6837a008f46 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-ffd9b320-1e56-4663-9f07-d6837a008f46');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ],
            "text/plain": [
              "         id diagnosis  ...  fractal_dimension_worst  Unnamed: 32\n",
              "0    842302         M  ...                  0.11890          NaN\n",
              "1    842517         M  ...                  0.08902          NaN\n",
              "2  84300903         M  ...                  0.08758          NaN\n",
              "3  84348301         M  ...                  0.17300          NaN\n",
              "4  84358402         M  ...                  0.07678          NaN\n",
              "\n",
              "[5 rows x 33 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "del data['Unnamed: 32']"
      ],
      "metadata": {
        "id": "PJvRHUZYQL8F"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ax = sns.countplot(data['diagnosis'], label= 'Count')\n",
        "B,M = data['diagnosis'].value_counts()\n",
        "print('Benign', B)\n",
        "print('Malignanat', M)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 368
        },
        "id": "xUX8eeNwQWsq",
        "outputId": "1b66fa73-9efd-4b12-8cd6-183b8db10408"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Benign 357\n",
            "Malignanat 212\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  FutureWarning\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEGCAYAAACKB4k+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAASDklEQVR4nO3df7BndX3f8efLBYWpJED2lm5216y1tAyauOgVSdI2BMeKpOmiQxyYSVwt0zUz2DFpJhNIO2psmWqDYaJJmFnKT2tU6o9CLLUhBHWcUXCh67KA1K1C2R1+XBEQQqSz67t/fD/349fL3eW7wLnfy97nY+bM95zP53PO932Zu/fF55zzPd9UFZIkAbxo2gVIkpYPQ0GS1BkKkqTOUJAkdYaCJKk7bNoFPBerV6+uDRs2TLsMSXpBufXWW79bVTOL9b2gQ2HDhg1s27Zt2mVI0gtKknv31+fpI0lSZyhIkjpDQZLUGQqSpM5QkCR1hoIkqTMUJEmdoSBJ6gwFSVL3gv5Es3Qo+78f+Nlpl6Bl6GXvvX3Q4w82U0hyRJJbknwjyR1J/qC1X5nkO0m2t2Vja0+SjyTZlWRHktcMVZskaXFDzhSeAk6rqieSHA58Jcn/aH2/W1WfXjD+zcDxbXk9cEl7lSQtkcFmCjXyRNs8vC0H+kLoTcDVbb+vAUcnWTNUfZKkpxv0QnOSVUm2Aw8BN1TVza3rwnaK6OIkL2lta4H7xnbf3doWHnNLkm1Jts3NzQ1ZviStOIOGQlXtq6qNwDrg5CSvAi4ATgBeBxwL/N5BHnNrVc1W1ezMzKKPA5ckPUtLcktqVT0K3AScXlX3t1NETwFXACe3YXuA9WO7rWttkqQlMuTdRzNJjm7rRwJvBL45f50gSYAzgZ1tl+uAt7e7kE4BHquq+4eqT5L0dEPefbQGuCrJKkbhc01VfT7JXyeZAQJsB36zjb8eOAPYBTwJvHPA2iRJixgsFKpqB3DSIu2n7Wd8AecNVY8k6Zn5mAtJUmcoSJI6Q0GS1BkKkqTOUJAkdYaCJKkzFCRJnaEgSeoMBUlSZyhIkjpDQZLUGQqSpM5QkCR1hoIkqTMUJEmdoSBJ6gwFSVJnKEiSOkNBktQZCpKkbrBQSHJEkluSfCPJHUn+oLW/PMnNSXYl+VSSF7f2l7TtXa1/w1C1SZIWN+RM4SngtKp6NbAROD3JKcCHgIur6h8AjwDntvHnAo+09ovbOEnSEhosFGrkibZ5eFsKOA34dGu/CjizrW9q27T+NyTJUPVJkp5u0GsKSVYl2Q48BNwA/B/g0ara24bsBta29bXAfQCt/zHgpxY55pYk25Jsm5ubG7J8SVpxBg2FqtpXVRuBdcDJwAnPwzG3VtVsVc3OzMw85xolST+yJHcfVdWjwE3AzwNHJzmsda0D9rT1PcB6gNb/k8DDS1GfJGlkyLuPZpIc3daPBN4I3MUoHM5qwzYD17b169o2rf+vq6qGqk+S9HSHPfOQZ20NcFWSVYzC55qq+nySO4FPJvkPwP8CLmvjLwM+lmQX8D3g7AFrkyQtYrBQqKodwEmLtH+b0fWFhe0/AH5tqHokSc/MTzRLkjpDQZLUGQqSpM5QkCR1hoIkqTMUJEmdoSBJ6gwFSVJnKEiSOkNBktQZCpKkzlCQJHWGgiSpMxQkSZ2hIEnqDAVJUmcoSJI6Q0GS1BkKkqTOUJAkdYOFQpL1SW5KcmeSO5K8p7W/P8meJNvbcsbYPhck2ZXk7iRvGqo2SdLiDhvw2HuB36mq25IcBdya5IbWd3FVXTQ+OMmJwNnAK4GfBv4qyT+sqn0D1ihJGjPYTKGq7q+q29r648BdwNoD7LIJ+GRVPVVV3wF2AScPVZ8k6emW5JpCkg3AScDNrendSXYkuTzJMa1tLXDf2G67WSREkmxJsi3Jtrm5uQGrlqSVZ/BQSPJS4DPAb1XV94FLgFcAG4H7gQ8fzPGqamtVzVbV7MzMzPNeryStZIOGQpLDGQXCx6vqswBV9WBV7auqHwKX8qNTRHuA9WO7r2ttkqQlMuTdRwEuA+6qqj8aa18zNuwtwM62fh1wdpKXJHk5cDxwy1D1SZKebsi7j34R+A3g9iTbW9vvA+ck2QgUcA/wLoCquiPJNcCdjO5cOs87jyRpaQ0WClX1FSCLdF1/gH0uBC4cqiZJ0oH5iWZJUmcoSJI6Q0GS1BkKkqTOUJAkdYaCJKkzFCRJnaEgSeoMBUlSZyhIkjpDQZLUGQqSpM5QkCR1hoIkqTMUJEmdoSBJ6ob85rUXhNf+7tXTLkHL0K1/+PZplyBNhTMFSVJnKEiSuolCIcmNk7RJkl7YDhgKSY5IciywOskxSY5tywZg7TPsuz7JTUnuTHJHkve09mOT3JDkW+31mNaeJB9JsivJjiSveX5+REnSpJ5ppvAu4FbghPY6v1wL/Mkz7LsX+J2qOhE4BTgvyYnA+cCNVXU8cGPbBngzcHxbtgCXHPRPI0l6Tg5491FV/THwx0n+dVV99GAOXFX3A/e39ceT3MVodrEJOLUNuwr4IvB7rf3qqirga0mOTrKmHUeStAQmuiW1qj6a5BeADeP7VNVE93O2000nATcDx439oX8AOK6trwXuG9ttd2v7sVBIsoXRTIKXvexlk7y9JGlCE4VCko8BrwC2A/tacwHPGApJXgp8Bvitqvp+kt5XVZWkDqbgqtoKbAWYnZ09qH0lSQc26YfXZoET26mdiSU5nFEgfLyqPtuaH5w/LZRkDfBQa98DrB/bfV1rkyQtkUk/p7AT+HsHc+CMpgSXAXdV1R+NdV0HbG7rmxldtJ5vf3u7C+kU4DGvJ0jS0pp0prAauDPJLcBT841V9S8OsM8vAr8B3J5ke2v7feCDwDVJzgXuBd7W+q4HzgB2AU8C75z0h5AkPT8mDYX3H+yBq+orQPbT/YZFxhdw3sG+jyTp+TPp3UdfGroQSdL0TXr30eOM7jYCeDFwOPA3VfUTQxUmSVp6k84UjppfbxeQNzH6lLIk6RBy0E9JrZH/BrxpgHokSVM06emjt45tvojR5xZ+MEhFkqSpmfTuo18dW98L3MPoFJIk6RAy6TUFPzMgSSvApF+ysy7J55I81JbPJFk3dHGSpKU16YXmKxg9huKn2/IXrU2SdAiZNBRmquqKqtrbliuBmQHrkiRNwaSh8HCSX0+yqi2/Djw8ZGGSpKU3aSj8S0YPrnuA0ZfenAW8Y6CaJElTMuktqR8ANlfVIwBJjgUuYhQWkqRDxKQzhZ+bDwSAqvoeo6/XlCQdQiYNhRclOWZ+o80UJp1lSJJeICb9w/5h4KtJ/mvb/jXgwmFKkiRNy6SfaL46yTbgtNb01qq6c7iyJEnTMPEpoBYCBoEkHcIO+tHZkqRDl6EgSeoGC4Ukl7eH5+0ca3t/kj1JtrfljLG+C5LsSnJ3Er/AR5KmYMiZwpXA6Yu0X1xVG9tyPUCSE4GzgVe2ff4syaoBa5MkLWKwUKiqLwPfm3D4JuCTVfVUVX0H2AWcPFRtkqTFTeOawruT7Ginl+Y/ELcWuG9szO7W9jRJtiTZlmTb3Nzc0LVK0oqy1KFwCfAKYCOjB+t9+GAPUFVbq2q2qmZnZnx6tyQ9n5Y0FKrqwaraV1U/BC7lR6eI9gDrx4aua22SpCW0pKGQZM3Y5luA+TuTrgPOTvKSJC8HjgduWcraJEkDPtQuySeAU4HVSXYD7wNOTbIRKOAe4F0AVXVHkmsYfWJ6L3BeVe0bqjZJ0uIGC4WqOmeR5ssOMP5CfMieJE2Vn2iWJHWGgiSpMxQkSZ2hIEnqDAVJUmcoSJI6Q0GS1BkKkqTOUJAkdYaCJKkzFCRJnaEgSeoMBUlSZyhIkjpDQZLUGQqSpM5QkCR1hoIkqTMUJEmdoSBJ6gYLhSSXJ3koyc6xtmOT3JDkW+31mNaeJB9JsivJjiSvGaouSdL+DTlTuBI4fUHb+cCNVXU8cGPbBngzcHxbtgCXDFiXJGk/BguFqvoy8L0FzZuAq9r6VcCZY+1X18jXgKOTrBmqNknS4pb6msJxVXV/W38AOK6trwXuGxu3u7U9TZItSbYl2TY3NzdcpZK0Ak3tQnNVFVDPYr+tVTVbVbMzMzMDVCZJK9dSh8KD86eF2utDrX0PsH5s3LrWJklaQksdCtcBm9v6ZuDasfa3t7uQTgEeGzvNJElaIocNdeAknwBOBVYn2Q28D/ggcE2Sc4F7gbe14dcDZwC7gCeBdw5VlyRp/wYLhao6Zz9db1hkbAHnDVWLJGkyfqJZktQZCpKkzlCQJHWGgiSpMxQkSZ2hIEnqDAVJUmcoSJI6Q0GS1BkKkqTOUJAkdYaCJKkzFCRJnaEgSeoMBUlSZyhIkjpDQZLUGQqSpM5QkCR1hoIkqTtsGm+a5B7gcWAfsLeqZpMcC3wK2ADcA7ytqh6ZRn2StFJNc6bwy1W1sapm2/b5wI1VdTxwY9uWJC2h5XT6aBNwVVu/CjhzirVI0oo0rVAo4C+T3JpkS2s7rqrub+sPAMcttmOSLUm2Jdk2Nze3FLVK0ooxlWsKwD+uqj1J/i5wQ5JvjndWVSWpxXasqq3AVoDZ2dlFx0iSnp2pzBSqak97fQj4HHAy8GCSNQDt9aFp1CZJK9mSh0KSv5PkqPl14J8BO4HrgM1t2Gbg2qWuTZJWummcPjoO+FyS+ff/86r6QpKvA9ckORe4F3jbFGqTpBVtyUOhqr4NvHqR9oeBNyx1PZKkH1lOt6RKkqbMUJAkdYaCJKkzFCRJnaEgSeoMBUlSZyhIkjpDQZLUGQqSpM5QkCR1hoIkqTMUJEmdoSBJ6gwFSVJnKEiSOkNBktQZCpKkzlCQJHWGgiSpMxQkSd2yC4Ukpye5O8muJOdPux5JWkmWVSgkWQX8KfBm4ETgnCQnTrcqSVo5llUoACcDu6rq21X1/4BPApumXJMkrRiHTbuABdYC941t7wZePz4gyRZgS9t8IsndS1TbSrAa+O60i1gOctHmaZegH+fv5rz35fk4ys/sr2O5hcIzqqqtwNZp13EoSrKtqmanXYe0kL+bS2e5nT7aA6wf217X2iRJS2C5hcLXgeOTvDzJi4GzgeumXJMkrRjL6vRRVe1N8m7gfwKrgMur6o4pl7WSeFpOy5W/m0skVTXtGiRJy8RyO30kSZoiQ0GS1BkKK1ySSvJfxrYPSzKX5PPTrEsCSLIvyfYk30hyW5JfmHZNh7pldaFZU/E3wKuSHFlVfwu8EW8D1vLxt1W1ESDJm4D/CPzSdEs6tDlTEMD1wK+09XOAT0yxFml/fgJ4ZNpFHOoMBcHoGVNnJzkC+Dng5inXI807sp0++ibwn4F/P+2CDnWePhJVtSPJBkazhOunW430Y8ZPH/08cHWSV5X30g/GmYLmXQdchKeOtExV1VcZPRhvZtq1HMqcKWje5cCjVXV7klOnXYy0UJITGD3p4OFp13IoMxQEQFXtBj4y7TqkBY5Msr2tB9hcVfumWdChzsdcSJI6rylIkjpDQZLUGQqSpM5QkCR1hoIkqfOWVKlJ8n7gCUbP2PlyVf3VFGv5wLRr0MpkKEgLVNV7rUErlaePtKIl+bdJ/neSrwD/qLVdmeSstv7eJF9PsjPJ1iRp7a9LsqM9rO0Pk+xs7e9I8tkkX0jyrST/aey9zklyezvWh1rbqvZ+O1vfby9SwweT3Nne76Il/Q+kFceZglasJK8FzgY2Mvq3cBtw64Jhf1JVH2jjPwb8c+AvgCuAf1VVX03ywQX7bAROAp4C7k7yUWAf8CHgtYwe//yXSc4E7gPWVtWr2nscvaDGnwLeApxQVbWwX3q+OVPQSvZPgM9V1ZNV9X1GDwVc6JeT3JzkduA04JXtD/NR7QFtAH++YJ8bq+qxqvoBcCfwM8DrgC9W1VxV7QU+DvxT4NvA30/y0SSnA99fcKzHgB8AlyV5K/Dkc/6ppQMwFKT9aN8v8WfAWVX1s8ClwBET7PrU2Po+DjAjr6pHgFcDXwR+k9F3Boz37wVOBj7NaJbyhcl/AungGQpayb4MnJnkyCRHAb+6oH8+AL6b5KXAWQBV9SjweJLXt/6zJ3ivW4BfSrI6ySpG313xpSSrgRdV1WeAfwe8Znyn9r4/WVXXA7/NKECkwXhNQStWVd2W5FPAN4CHgK8v6H80yaXATuCBBf3nApcm+SHwJUaneQ70XvcnOR+4idHTPv97VV2b5NXAFUnm/wftggW7HgVc22YtAf7Ns/hRpYn5lFTpWUjy0qp6oq2fD6ypqvdMuSzpOXOmID07v5LkAkb/hu4F3jHdcqTnhzMFSVLnhWZJUmcoSJI6Q0GS1BkKkqTOUJAkdf8f0rm+gk1Pwo0AAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = data.iloc[:, 2:].values\n",
        "y = data.iloc[:, 1].values"
      ],
      "metadata": {
        "id": "iGEvSWKGQiDn"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Encoding categorical data\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "labelencoder_X = LabelEncoder()\n",
        "y = labelencoder_X.fit_transform(y)\n",
        "\n",
        "# Splitting the dataset into the Training set and Test set\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)\n",
        "\n",
        "#Feature Scaling\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "sc = StandardScaler()\n",
        "X_train = sc.fit_transform(X_train)\n",
        "X_test = sc.transform(X_test)\n",
        "\n",
        "X_train\n",
        "\n",
        "X_test"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D1WX4LR5RC5O",
        "outputId": "44c3516f-fd8e-4d51-fbe6-eb54aaf92eab"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-0.20175604,  0.3290786 , -0.13086754, ...,  1.3893291 ,\n",
              "         1.08203284,  1.54029664],\n",
              "       [-0.25555773,  1.46763319, -0.31780437, ..., -0.83369364,\n",
              "        -0.73131577, -0.87732522],\n",
              "       [-0.02619262, -0.8407682 , -0.09175081, ..., -0.49483785,\n",
              "        -1.22080864, -0.92115937],\n",
              "       ...,\n",
              "       [ 1.71811488,  0.09318356,  1.7286186 , ...,  1.57630515,\n",
              "         0.20317063, -0.15406178],\n",
              "       [ 1.18859296,  0.34352115,  1.19333694, ...,  0.56019755,\n",
              "         0.26991966, -0.27320074],\n",
              "       [ 0.26263752, -0.58080224,  0.28459338, ..., -0.19383705,\n",
              "        -1.15564888,  0.11231497]])"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Creating model \n",
        "def built_model():\n",
        "  model = Sequential()\n",
        "  #First layer\n",
        "  model.add(Dense(output_dim=16, init='uniform', activation='relu',input_dim=30))\n",
        "  #Second layer\n",
        "  model.add(Dense(output_dim=16, init='uniform', activation='relu'))\n",
        "  #Output layer\n",
        "  model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))\n",
        "  #compiling the model\n",
        "  model.compile(optimizer=\"Adam\", loss='binary_crossentropy', metrics=['accuracy'])\n",
        "  return model"
      ],
      "metadata": {
        "id": "daxJf9C6RYD6"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = KerasClassifier(build_fn = built_model, batch_size = 100, epochs=100)\n",
        "accuracies = cross_val_score(estimator = model, X = X_train, y=y_train, cv=10, n_jobs =-1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ayZgPcCJSZM3",
        "outputId": "7b0e555b-d1d7-4f22-f630-d9c37290ce28"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead.\n",
            "  \"\"\"Entry point for launching an IPython kernel.\n",
            "/usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_validation.py:372: FitFailedWarning: \n",
            "10 fits failed out of a total of 10.\n",
            "The score on these train-test partitions for these parameters will be set to nan.\n",
            "If these failures are not expected, you can try to debug them by setting error_score='raise'.\n",
            "\n",
            "Below are more details about the failures:\n",
            "--------------------------------------------------------------------------------\n",
            "10 fits failed with the following error:\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_validation.py\", line 680, in _fit_and_score\n",
            "    estimator.fit(X_train, y_train, **fit_params)\n",
            "  File \"/usr/local/lib/python3.7/dist-packages/keras/wrappers/scikit_learn.py\", line 232, in fit\n",
            "    return super(KerasClassifier, self).fit(x, y, **kwargs)\n",
            "  File \"/usr/local/lib/python3.7/dist-packages/keras/wrappers/scikit_learn.py\", line 155, in fit\n",
            "    self.model = self.build_fn(**self.filter_sk_params(self.build_fn))\n",
            "  File \"<ipython-input-19-23a6b098f510>\", line 5, in built_model\n",
            "TypeError: __init__() missing 1 required positional argument: 'units'\n",
            "\n",
            "  warnings.warn(some_fits_failed_message, FitFailedWarning)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "accuracies\n",
        "accuracies.mean()\n",
        "accuracies.std()"
      ],
      "metadata": {
        "id": "nnEAHtHLTdVK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "bhrb30_wTxJy"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}