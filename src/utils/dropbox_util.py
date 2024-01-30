import os

import dropbox
from dropbox.files import WriteMode
from tqdm import tqdm


def check_dropbox_file_exists(path, base_folder = "probing-results"):
    dbx = dropbox.Dropbox(app_key="APP_KEY", app_secret="APP_SECRET", oauth2_refresh_token=DROPBOX_TOKEN, timeout=30000)
    dbx.refresh_access_token()
    try:
        metadata = dbx.files_get_metadata("/" + base_folder + "/"+path)
    except:
        return False
    return True

def download_artifacts(client, run, artifact_dir, local_dir):
    return client.download_artifacts(run.info.run_id, artifact_dir, local_dir)

def download_dropbox_artifacts(task, fold, seed, task_type, path):

    zip_file = "run_" + task + "_" + str(fold) + "_" + str(seed) + "_" + task_type + "_bert-base-uncased.tar.gz"

    dbx = dropbox.Dropbox(app_key="APP_KEY", app_secret="APP_SECRET", oauth2_refresh_token=DROPBOX_TOKEN, timeout=30000)
    dbx.refresh_access_token()
    try:
        dbx.files_download_to_file(path + "/" + zip_file, "/probing/" + zip_file)

        os.system("cd " + path + "; mkdir artifacts")
        os.system("cd " + path + "; tar -xvf " + zip_file)
        os.system("cd " + path + "; find . -maxdepth 3 -name \"models\" -exec mv {} ./artifacts \;")
        os.system("cd " + path + "; find . -maxdepth 3 -name \"preds\" -exec mv {} ./artifacts \;")
    except:
        print("exists")

    os.system("cd " + path + "; rm -rf " + zip_file.replace(".tar.gz", ""))
    os.system("cd " + path + "; rm " + zip_file)


def download_dropbox_file(source_path, dest_path, file):

    dbx = dropbox.Dropbox(app_key="APP_KEY", app_secret="APP_SECRET", oauth2_refresh_token=DROPBOX_TOKEN, timeout=30000)
    dbx.refresh_access_token()
    dbx.files_download_to_file(dest_path + file, source_path + file)

    return dest_path

def list_dropbox_files(path):

    dbx = dropbox.Dropbox(app_key="APP_KEY", app_secret="APP_SECRET", oauth2_refresh_token=DROPBOX_TOKEN, timeout=30000)
    dbx.refresh_access_token()
    return dbx.files_list_folder(path)



def upload(
        file_path,
        target_path,
        timeout=3000,
        chunk_size=4 * 1024 * 1024,
):
    dbx = dropbox.Dropbox(app_key="APP_KEY", app_secret="APP_SECRET", oauth2_refresh_token=DROPBOX_TOKEN, timeout=timeout)
    dbx.refresh_access_token()
    try:
        dbx.files_delete_v2(target_path)
    except:
        print("done")

    with open(file_path, "rb") as f:
        file_size = os.path.getsize(file_path)
        chunk_size = 32 * 1024 * 1024
        if file_size <= chunk_size:
            print(dbx.files_upload(f.read(), target_path, mode=dropbox.files.WriteMode.overwrite))
        else:
            with tqdm(total=file_size, desc="Uploaded") as pbar:
                upload_session_start_result = dbx.files_upload_session_start(
                    f.read(chunk_size)
                )
                pbar.update(chunk_size)
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=f.tell()
                )
                commit = dropbox.files.CommitInfo(path=target_path, mode=dropbox.files.WriteMode.overwrite)
                while f.tell() < file_size:
                    if (file_size - f.tell()) <= chunk_size:
                        print(
                            dbx.files_upload_session_finish(
                                f.read(chunk_size), cursor, commit
                            )
                        )
                    else:
                        dbx.files_upload_session_append(
                            f.read(chunk_size),
                            cursor.session_id,
                            cursor.offset,
                        )
                        cursor.offset = f.tell()
                    pbar.update(chunk_size)

