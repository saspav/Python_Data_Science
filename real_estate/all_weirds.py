from process_data import *

files_path = os.path.join(PATH_EXPORT, 'best')
weird = pd.DataFrame()
for curr_dir, dirs, files in os.walk(files_path):
    for file in files:
        file_no_ext = os.path.splitext(file)[0]
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(curr_dir, file))
            # Id,Price
            df.columns = ['Id', file_no_ext]
            if not len(weird):
                weird = df
            else:
                weird = weird.merge(df, on='Id')

weird = weird.set_index('Id')
weird['Price'] = weird.mean(axis=1)
submission = pd.DataFrame({
    'Id': weird.index,
    'Price': weird.Price
})
submission.to_csv(os.path.join(PATH_EXPORT, 'submit_weird.csv'), index=False)
