from fast_keywords.objects import keywords, trainer
import pandas as pd
import dill
import re


LANGUAGE = 'german'
WORDLIST = "Suchworte.xlsx"
PREFIX = "fast_keywords/res/nielsen/"

def main():
    words = pd.read_excel(f"{PREFIX}{WORDLIST}")
    kw = keywords.Keywords(words.searchtext.tolist(), ids=words.id.tolist())
    output = pd.read_csv('OUTPUT.csv').infer_objects()
    for name, group in output.groupby(["Keyword"]):
        try:
            if group["Match is Invalid"].astype('int32').any():
                environment_vectors = []
                environment_vector_labels = []
                for i, (_, row) in enumerate(group.iterrows()):
                    environment = row["Surrounding Text"]
                    # Remove capital letters ie the original entity.
                    environment = re.sub(r'[A-Z]', '', environment)
                    environment_vectors.append(kw.get_vector(environment).toarray()[0])
                    environment_vector_labels.append(int(row['Match is Invalid']))

                try:
                    # Try to load a pre-existing model.
                    with open(f"fast_keywords/models/{LANGUAGE}/{name}", "rb") as f:
                        model = dill.load(f)

                    # Append data and labels to model.
                    model.data.extend(environment_vectors)
                    model.labels.extend(environment_vector_labels)

                except FileNotFoundError:
                    # If does not exist, create new trainer object.
                    model = trainer.Trainer(
                            language=LANGUAGE,
                            keyword=name,
                            data=environment_vectors,
                            labels=environment_vector_labels,
                        )

                # Fit the trainer object to the updated data for that word.
                model.train()
                # Save the fitted object to models dir for use during runtime.
                with open (f"fast_keywords/models/{LANGUAGE}/{name}", "wb") as f:
                    dill.dump(model, f)

        except ValueError:
            pass


if __name__ == '__main__':
    main()


# TODO: fully clarify the workflow ... if they train a model how will this effect new data? we want to
# avoid overfitting but allow for retraining etc ...
# trainer class is robust enough that you can easily continue training on newly-labeled csv
# TODO: some errors in the csv formatting because columns are getting fucked up
# TODO: Issue with variable assignment in environment method?
