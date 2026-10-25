# Notebooks (Google Colab)

This folder holds the exploratory data analysis (EDA) notebooks. They are
written and run in **Google Colab**. All other project code lives in `src/`
as normal Python files and runs locally.

## How to work with these notebooks

1. Run the cleaning code locally (PR 2) to create the files in
   `data/processed/sgcc/` and `data/processed/oedi/`.
2. Upload those processed files to a shared Google Drive folder.
3. In Colab, connect Google Drive and read the files from there:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

4. When the notebook is ready, save it to this folder in GitHub with
   **File → Save a copy in GitHub** in Colab. Save to a new branch and
   open a pull request, like any other change.

## Rules

- Use the **processed** data, not the raw downloads, so the plots match
  what the models see.
- Write a short explanation under each plot (what it shows and why it
  matters).
- Do not use the **test** set to make decisions; explore the training
  data only.
