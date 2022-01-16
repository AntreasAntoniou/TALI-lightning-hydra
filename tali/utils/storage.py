"""
Storage associated utilities
"""
import json
import logging as log
import os
import os.path
import pathlib
from dataclasses import dataclass
from typing import Dict

import requests
import tqdm  # progress bar
import yaml
from google.cloud import storage
from omegaconf import DictConfig

from tali.utils.arg_parsing import DictWithDotNotation


def build_experiment_folder(experiment_name, log_path, save_images=True):
    """
    An experiment logging folder goes along with each  This builds that
    folder
    :param args: dictionary of arguments
    :return: filepaths for saved models, logs, and images
    """
    saved_models_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "saved_models"
    )
    logs_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "summary_logs"
    )
    images_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "images"
    )

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)

    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    if not os.path.exists(images_filepath):
        os.makedirs(images_filepath)

    if save_images:
        if not os.path.exists(images_filepath + "/train"):
            os.makedirs(images_filepath + "/train")
        if not os.path.exists(images_filepath + "/val"):
            os.makedirs(images_filepath + "/val")
        if not os.path.exists(images_filepath + "/test"):
            os.makedirs(images_filepath + "/test")

    return saved_models_filepath, logs_filepath, images_filepath


def download_file_from_url(url, filename=None, verbose=False):
    """
    Download file with progressbar
    __author__ = "github.com/ruxi"
    __license__ = "MIT"
    Usage:
        download_file('http://web4host.net/5MB.zip')
    """
    if not filename:
        local_filename = os.path.join("", url.split("/")[-1])
    else:
        local_filename = filename
    r = requests.get(url, stream=True)
    file_size = int(r.headers["Content-Length"])
    chunk_size = 1024
    num_bars = file_size // chunk_size
    if verbose:
        log.info(dict(file_size=file_size))
        log.info(dict(num_bars=num_bars))

    file_directory = filename.replace(filename.split("/")[-1], "")

    file_directory = pathlib.Path(file_directory)

    file_directory.mkdir(parents=True, exist_ok=True)

    with open(local_filename, "wb") as fp:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=num_bars,
            unit="KB",
            desc=local_filename,
            leave=True,  # progressbar stays
        ):
            fp.write(chunk)
    return local_filename


def save_json(filepath, metrics_dict, overwrite):
    """
    Saves a metrics .json file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param metrics_dict: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath,
    if False adds metrics to existing
    """

    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)

    if ".json" not in filepath:
        filepath = f"{filepath}.json"

    metrics_file_path = filepath

    if overwrite and os.path.exists(metrics_file_path):
        os.remove(metrics_file_path)

    parent_folder = "/".join(filepath.split("/")[:-1])
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)

    # 'r'       open for reading (default)
    # 'w'       open for writing, truncating the file first
    # 'x'       create a new file and open it for writing
    # 'a'       open for writing, appending to the end of the file if it exists
    # 'b'       binary mode
    # 't'       text mode (default)
    # '+'       open a disk file for updating (reading and writing)
    # 'U'       universal newline mode (deprecated)

    with open(metrics_file_path, "w+") as json_file:
        json.dump(metrics_dict, json_file, indent=4, sort_keys=True)


def load_json(filepath):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)

    if ".json" not in filepath:
        filepath = f"{filepath}.json"

    with open(filepath) as json_file:
        metrics_dict = json.load(json_file)

    return metrics_dict


def save_model_config_to_yaml(model_name, config_attributes, config_filepath):
    config_filepath = pathlib.Path(config_filepath)
    try:
        with open(config_filepath, mode="w+") as file_reader:
            config = yaml.safe_load(file_reader)
            config[model_name] = config_attributes
            yaml.dump(config, file_reader)
            return True
    except Exception:
        log.exception("Could not save model config to yaml")
        return False


def load_model_config_from_yaml(config_filepath, model_name):
    config_filepath = pathlib.Path(config_filepath)
    with open(config_filepath, mode="r") as file_reader:
        config = yaml.safe_load(file_reader)

    return DictWithDotNotation(config[model_name])


def load_yaml(filepath):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if isinstance(filepath, pathlib.Path):
        filepath = os.fspath(filepath.resolve())

    if ".yaml" not in filepath:
        filepath = f"{filepath}.yaml"

    with open(filepath) as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    return config_dict


def pretty_print_dict(input_dict, tab=0):
    output_string = []
    current_tab = tab
    tab_string = "".join(current_tab * ["\t"])
    for key, value in input_dict.items():

        # logging.info(f'{key} is {type(value)}')

        if isinstance(value, (Dict, DictConfig)):
            # logging.info(f'{key} is Dict')
            value = pretty_print_dict(value, tab=current_tab + 1)

        output_string.append(f"\n{tab_string}{key}: {value}")
    return "".join(output_string)


def create_bucket_class_location(bucket_name, storage_class, location):
    """
    Create a new bucket in the US region with the coldline storage
    class
    """
    # bucket_name = "your-new-bucket-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = storage_class
    new_bucket = storage_client.create_bucket(bucket, location=location)

    log.info(
        f"Created bucket {new_bucket.name} "
        f"in {new_bucket.location} "
        f"with storage class {new_bucket.storage_class}"
    )
    return new_bucket


def get_bucket_client(bucket_name):
    storage_client = storage.Client()
    bucket_client = storage_client.bucket(bucket_name)

    return storage_client, bucket_client


def upload_file(
    bucket_name, source_file_name, destination_file_name, bucket_client=None
):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    if bucket_client is None:
        storage_client = storage.Client()
        bucket_client = storage_client.bucket(bucket_name)

    blob = bucket_client.blob(destination_file_name)

    blob.upload_from_filename(source_file_name)

    log.info(f"File {source_file_name} uploaded to {destination_file_name}")


def download_file(
    bucket_name, source_file_name, destination_file_name, bucket_client=None
):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    if bucket_client is None:
        storage_client = storage.Client()
        bucket_client = storage_client.bucket(bucket_name)
    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket_client.blob(source_file_name)
    blob.download_to_filename(destination_file_name)

    log.info(
        f"Downloaded storage object {source_file_name} "
        f"from bucket {bucket_name} "
        f"to local file {destination_file_name}"
    )


@dataclass
class GoogleStorageClientConfig:
    local_log_dir: str
    bucket_name: str
    experiment_name: str


# Use cases: 1. (New) Brand new experiment: It should scan the directory structure as
# it is, delete whatever is on google storage and upload the new structure and keep
# updating it periodically. - rsync with 'syncing local to remote' from local,
# then in subsequent updates just do default rsync 2. (Continued-full or
# Continued-minimal) Continued experiment: It should find the online experiment
# files, compare with local files and download the missing files OR just download
# necessary files to continue - Continued full, full rsync from bucket to local,
# continued in all updates - Continued minimal, full rsync from bucket to local
# excluding a prefix or postfix that ensures all weights other than 'latest' are not
# synced

#


def google_storage_rsync_gs_to_local(
    bucket_name, experiments_root_dir, experiment_name, exclude_list, options_list
):
    # NAME
    #   rsync - Synchronize content of two buckets/directories
    #
    #
    # SYNOPSIS
    #
    #   gsutil rsync [OPTION]... src_url dst_url
    #
    #
    #
    # DESCRIPTION
    #   The gsutil rsync command makes the contents under dst_url the same as the
    #   contents under src_url, by copying any missing files/objects (or those whose
    #   data has changed), and (if the -d option is specified) deleting any extra
    #   files/objects. src_url must specify a directory, bucket, or bucket
    #   subdirectory. For example, to sync the contents of the local directory "data"
    #   to the bucket gs://mybucket/data, you could do:
    #
    #     gsutil rsync data gs://mybucket/data
    #
    #   To recurse into directories use the -r option:
    #
    #     gsutil rsync -r data gs://mybucket/data
    #
    #   If you have a large number of objects to synchronize you might want to use the
    #   gsutil option (see "gsutil help options"), to perform parallel
    #   (multi-threaded/multi-processing) synchronization:
    #
    #     gsutil rsync -r data gs://mybucket/data
    #
    #   The -m option typically will provide a large performance boost if either the
    #   source or destination (or both) is a cloud URL. If both source and
    #   destination are file URLs the -m option will typically thrash the disk and
    #   slow synchronization down.
    #
    #   Note 1: Shells (like bash, zsh) sometimes attempt to expand wildcards in ways
    #   that can be surprising. Also, attempting to copy files whose names contain
    #   wildcard characters can result in problems. For more details about these
    #   issues see the section "POTENTIALLY SURPRISING BEHAVIOR WHEN USING WILDCARDS"
    #   under "gsutil help wildcards".
    #
    #   Note 2: If you are synchronizing a large amount of data between clouds you
    #   might consider setting up a
    #   `Google Compute Engine <https://cloud.google.com/products/compute-engine>`_
    #   account and running gsutil there. Since cross-provider gsutil data transfers
    #   flow through the machine where gsutil is running, doing this can make your
    #   transfer run significantly faster than running gsutil on your local
    #   workstation.
    #
    #   Note 3: rsync does not copy empty directory trees, since Cloud Storage uses a
    #   `flat namespace <https://cloud.google.com/storage/docs/folders>`_.
    #
    #
    # Using -d Option (with caution!) to mirror source and destination.
    #   The rsync -d option is very useful and commonly used, because it provides a
    #   means of making the contents of a destination bucket or directory match those
    #   of a source bucket or directory. This is done by copying all data from the
    #   source to the destination and deleting all other data in the destination that
    #   is not in the source. Please exercise caution when you
    #   use this option: It's possible to delete large amounts of data accidentally
    #   if, for example, you erroneously reverse source and destination.
    #
    #   To make the local directory my-data the same as the contents of
    #   gs://mybucket/data and delete objects in the local directory that are not in
    #   gs://mybucket/data:
    #
    #     gsutil rsync -d -r gs://mybucket/data my-data
    #
    #   To make the contents of gs://mybucket2 the same as gs://mybucket1 and delete
    #   objects in gs://mybucket2 that are not in gs://mybucket1:
    #
    #     gsutil rsync -d -r gs://mybucket1 gs://mybucket2
    #
    # You can also mirror data across local directories. This example will copy all
    # objects from dir1 into dir2 and delete all objects in dir2 which are not in dir1:
    #
    #     gsutil rsync -d -r dir1 dir2
    #
    #   To mirror your content across clouds:
    #
    #     gsutil rsync -d -r gs://my-gs-bucket s3://my-s3-bucket
    #
    #   Change detection works if the other Cloud provider is using md5 or CRC32. AWS
    #   multipart upload has an incompatible checksum.
    #
    #   As mentioned above, using -d can be dangerous because of how quickly data can
    #   be deleted. For example, if you meant to synchronize a local directory from
    #   a bucket in the cloud but instead run the command:
    #
    #     gsutil rsync -r -d ./your-dir gs://your-bucket
    #
    #   and your-dir is currently empty, you will quickly delete all of the objects in
    #   gs://your-bucket.
    #
    #   You can also cause large amounts of data to be lost quickly by specifying a
    #   subdirectory of the destination as the source of an rsync. For example, the
    #   command:
    #
    #     gsutil rsync -r -d gs://your-bucket/data gs://your-bucket
    #
    #   would cause most or all of the objects in gs://your-bucket to be deleted
    #   (some objects may survive if there are any with names that sort lower than
    #   "data" under gs://your-bucket/data).
    #
    #   In addition to paying careful attention to the source and destination you
    #   specify with the rsync command, there are two more safety measures you can
    #   take when using gsutil rsync -d:
    #
    #   1. Try running the command with the rsync -n option first, to see what it
    #      would do without actually performing the operations. For example, if
    #      you run the command:
    #
    #        gsutil rsync -r -d -n gs://your-bucket/data gs://your-bucket
    #
    #      it will be immediately evident that running that command without the -n
    #      option would cause many objects to be deleted.
    #
    #   2. Enable object versioning in your bucket, which allows you to restore
    #      objects if you accidentally delete them. For more details see
    #      `Object Versioning
    #      <https://cloud.google.com/storage/docs/object-versioning>`_.
    #
    #
    # BE CAREFUL WHEN SYNCHRONIZING OVER OS-SPECIFIC FILE TYPES (SYMLINKS, DEVICES,
    # ETC.) Running gsutil rsync over a directory containing operating
    # system-specific file types (symbolic links, device files, sockets, named pipes,
    # etc.) can cause various problems. For example, running a command like:
    #
    #     gsutil rsync -r ./dir gs://my-bucket
    #
    #   will cause gsutil to follow any symbolic links in ./dir, creating objects in
    #   my-bucket containing the data from the files to which the symlinks point. This
    #   can cause various problems:
    #
    #   * If you use gsutil rsync as a simple way to backup a directory to a bucket,
    #     restoring from that bucket will result in files where the symlinks used
    #     to be. At best this is wasteful of space, and at worst it can result in
    #     outdated data or broken applications -- depending on what is consuming
    #     the symlinks.
    #
    #   * If you use gsutil rsync over directories containing broken symlinks,
    #     gsutil rsync will abort (unless you pass the -e option).
    #
    #   * gsutil rsync skips symlinks that point to directories.
    #
    #   Since gsutil rsync is intended to support data operations (like moving a data
    #   set to the cloud for computational processing) and it needs to be compatible
    #   both in the cloud and across common operating systems, there are no plans for
    #   gsutil rsync to support operating system-specific file types like symlinks.
    #
    #   We recommend that users do one of the following:
    #
    #   * Don't use gsutil rsync over directories containing symlinks or other OS-
    #     specific file types.
    #   * Use the -e option to exclude symlinks or the -x option to exclude
    #     OS-specific file types by name.
    #   * Use a tool (such as tar) that preserves symlinks and other OS-specific file
    #     types, packaging up directories containing such files before uploading to
    #     the cloud.
    #
    #
    # EVENTUAL CONSISTENCY WITH NON-GOOGLE CLOUD PROVIDERS
    #   While Google Cloud Storage is strongly consistent, some cloud providers
    #   only support eventual consistency. You may encounter scenarios where rsync
    #   synchronizes using stale listing data when working with these other cloud
    #   providers. For example, if you run rsync immediately after uploading an
    #   object to an eventually consistent cloud provider, the added object may not
    #   yet appear in the provider's listing. Consequently, rsync will miss adding
    #   the object to the destination. If this happens you can rerun the rsync
    #   operation again later (after the object listing has "caught up").
    #
    #
    #
    # CHECKSUM VALIDATION AND FAILURE HANDLING
    #   At the end of every upload or download, the gsutil rsync command validates
    #   that the checksum of the source file/object matches the checksum of the
    #   destination file/object. If the checksums do not match, gsutil deletes
    #   the invalid copy and print a warning message. This very rarely happens, but
    #   if it does, please contact gs-team@google.com.
    #
    #   The rsync command retries failures when it is useful to do so, but if
    #   enough failures happen during a particular copy or delete operation, or if
    #   a failure isn't retryable, the overall command fails.
    #
    #   If the -C option is provided, the command instead skips failing objects and
    #   moves on. At the end of the synchronization run, if any failures were not
    #   successfully retried, the rsync command reports the count of failures and
    #   exits with non-zero status. At this point you can run the rsync command
    #   again, and gsutil attempts any remaining needed copy and/or delete
    #   operations.
    #
    #   For more details about gsutil's retry handling, see `Retry strategy
    #   <https://cloud.google.com/storage/docs/retry-strategy#tools>`_.
    #
    #
    # CHANGE DETECTION ALGORITHM
    #   To determine if a file or object has changed, gsutil rsync first checks
    #   whether the file modification time (mtime) of both the source and destination
    #   is available. If mtime is available at both source and destination, and the
    #   destination mtime is different than the source, or if the source and
    #   destination file size differ, gsutil rsync will update the destination. If the
    #   source is a cloud bucket and the destination is a local file system, and if
    #   mtime is not available for the source, gsutil rsync will use the time created
    #   for the cloud object as a substitute for mtime. Otherwise, if mtime is not
    #   available for either the source or the destination, gsutil rsync will fall
    #   back to using checksums. If the source and destination are both cloud buckets
    #   with checksums available, gsutil rsync will use these hashes instead of mtime.
    #   However, gsutil rsync will still update mtime at the destination if it is not
    #   present. If the source and destination have matching checksums and only the
    #   source has an mtime, gsutil rsync will copy the mtime to the destination. If
    #   neither mtime nor checksums are available, gsutil rsync will resort to
    #   comparing file sizes.
    #
    #   Checksums will not be available when comparing composite Google Cloud Storage
    #   objects with objects at a cloud provider that does not support CRC32C (which
    #   is the only checksum available for composite objects). See 'gsutil help
    #   compose' for details about composite objects.
    #
    #
    # COPYING IN THE CLOUD AND METADATA PRESERVATION
    #   If both the source and destination URL are cloud URLs from the same provider,
    #   gsutil copies data "in the cloud" (i.e., without downloading to and uploading
    #   from the machine where you run gsutil). In addition to the performance and
    #   cost advantages of doing this, copying in the cloud preserves metadata (like
    #   Content-Type and Cache-Control). In contrast, when you download data from the
    #   cloud it ends up in a file, which has no associated metadata, other than file
    #   modification time (mtime). Thus, unless you have some way to hold on to or
    #   re-create that metadata, synchronizing a bucket to a directory in the local
    #   file system will not retain the metadata other than mtime.
    #
    #   Note that by default, the gsutil rsync command does not copy the ACLs of
    #   objects being synchronized and instead will use the default bucket ACL (see
    #   "gsutil help defacl"). You can override this behavior with the -p option (see
    #   OPTIONS below).
    #
    #
    # SLOW CHECKSUMS
    #   If you find that CRC32C checksum computation runs slowly, this is likely
    #   because you don't have a compiled CRC32c on your system. Try running:
    #
    #     gsutil ver -l
    #
    #   If the output contains:
    #
    #     compiled crcmod: False
    #
    #   you are running a Python library for computing CRC32C, which is much slower
    #   than using the compiled code. For information on getting a compiled CRC32C
    #   implementation, see 'gsutil help crc32c'.
    #
    #
    # LIMITATIONS
    #
    #   1. The gsutil rsync command will only allow non-negative file modification
    #      times to be used in its comparisons. This means gsutil rsync will resort to
    #      using checksums for any file with a timestamp before 1970-01-01 UTC.
    #
    #   2. The gsutil rsync command considers only the live object version in
    #      the source and destination buckets when deciding what to copy / delete. If
    #      versioning is enabled in the destination bucket then gsutil rsync's
    #      replacing or deleting objects will end up creating versions, but the
    #      command doesn't try to make any noncurrent versions match in the source
    #      and destination buckets.
    #
    #   3. The gsutil rsync command does not support copying special file types
    #      such as sockets, device files, named pipes, or any other non-standard
    #      files intended to represent an operating system resource. If you run
    #      gsutil rsync on a source directory that includes such files (for example,
    #      copying the root directory on Linux that includes /dev ), you should use
    #      the -x flag to exclude these files. Otherwise, gsutil rsync may fail or
    #      hang.
    #
    #   4. The gsutil rsync command copies changed files in their entirety and does
    #      not employ the
    #      `rsync delta-transfer algorithm <https://rsync.samba.org/tech_report/>`_
    #      to transfer portions of a changed file. This is because Cloud Storage
    #      objects are immutable and no facility exists to read partial object
    #      checksums or perform partial replacements.
    #
    # OPTIONS
    #   -a canned_acl  Sets named canned_acl when uploaded objects created. See
    #                  "gsutil help acls" for further details. Note that rsync will
    #                  decide whether or not to perform a copy based only on object
    #                  size and modification time, not current ACL state. Also see the
    #                  -p option below.
    #
    #   -c             Causes the rsync command to compute and compare checksums
    #                  (instead of comparing mtime) for files if the size of source
    #                  and destination match. This option increases local disk I/O and
    #                  run time if either src_url or dst_url are on the local file
    #                  system.
    #
    #   -C             If an error occurs, continue to attempt to copy the remaining
    #                  files. If errors occurred, gsutil's exit status will be
    #                  non-zero even if this flag is set. This option is implicitly
    #                  set when running "gsutil rsync...".
    #
    #                  NOTE: -C only applies to the actual copying operation. If an
    #                  error occurs while iterating over the files in the local
    #                  directory (e.g., invalid Unicode file name) gsutil will print
    #                  an error message and abort.
    #
    #   -d             Delete extra files under dst_url not found under src_url. By
    #                  default extra files are not deleted.
    #
    #                  NOTE: this option can delete data quickly if you specify the
    #                  wrong source/destination combination. See the help section
    #                  above, "BE CAREFUL WHEN USING -d OPTION!".
    #
    #   -e             Exclude symlinks. When specified, symbolic links will be
    #                  ignored. Note that gsutil does not follow directory symlinks,
    #                  regardless of whether -e is specified.
    #
    # -i             This forces rsync to skip any files which exist on the
    # destination and have a modified time that is newer than the source file. (If an
    # existing destination file has a modification time equal to the source file's,
    # it will be updated if the sizes are different.)
    #
    #   -j <ext,...>   Applies gzip transport encoding to any file upload whose
    #                  extension matches the -j extension list. This is useful when
    #                  uploading files with compressible content (such as .js, .css,
    #                  or .html files) because it saves network bandwidth while
    #                  also leaving the data uncompressed in Google Cloud Storage.
    #
    #                  When you specify the -j option, files being uploaded are
    #                  compressed in-memory and on-the-wire only. Both the local
    #                  files and Cloud Storage objects remain uncompressed. The
    #                  uploaded objects retain the Content-Type and name of the
    #                  original files.
    #
    #                  Note that if you want to use the top-level -m option to
    #                  parallelize copies along with the -j/-J options, your
    #                  performance may be bottlenecked by the
    #                  "max_upload_compression_buffer_size" boto config option,
    #                  which is set to 2 GiB by default. This compression buffer
    #                  size can be changed to a higher limit, e.g.:
    #
    #                    gsutil -o "GSUtil:max_upload_compression_buffer_size=8G" \
    #                      -m rsync -j html,txt /local/source/dir gs://bucket/path
    #
    #   -J             Applies gzip transport encoding to file uploads. This option
    #                  works like the -j option described above, but it applies to
    #                  all uploaded files, regardless of extension.
    #
    #                  CAUTION: If you use this option and some of the source files
    #                  don't compress well (e.g., that's often true of binary data),
    #                  this option may result in longer uploads.
    #
    #   -n             Causes rsync to run in "dry run" mode, i.e., just outputting
    #                  what would be copied or deleted without actually doing any
    #                  copying/deleting.
    #
    #   -p             Causes ACLs to be preserved when objects are copied. Note that
    #                  rsync will decide whether or not to perform a copy based only
    #                  on object size and modification time, not current ACL state.
    #                  Thus, if the source and destination differ in size or
    #                  modification time and you run gsutil rsync -p, the file will be
    #                  copied and ACL preserved. However, if the source and
    #                  destination don't differ in size or checksum but have different
    #                  ACLs, running gsutil rsync -p will have no effect.
    #
    #                  Note that this option has performance and cost implications
    #                  when using the XML API, as it requires separate HTTP calls for
    #                  interacting with ACLs. The performance issue can be mitigated
    #                  to some degree by using gsutil rsync to cause parallel
    #                  synchronization. Also, this option only works if you have OWNER
    #                  access to all of the objects that are copied.
    #
    #                  You can avoid the additional performance and cost of using
    #                  rsync -p if you want all objects in the destination bucket to
    #                  end up with the same ACL by setting a default object ACL on
    #                  that bucket instead of using rsync -p. See 'gsutil help
    #                  defacl'.
    #
    #   -P             Causes POSIX attributes to be preserved when objects are
    #                  copied.  With this feature enabled, gsutil rsync will copy
    #                  fields provided by stat. These are the user ID of the owner,
    #                  the group ID of the owning group, the mode (permissions) of the
    #                  file, and the access/modification time of the file. For
    #                  downloads, these attributes will only be set if the source
    #                  objects were uploaded with this flag enabled.
    #
    #                  On Windows, this flag will only set and restore access time and
    #                  modification time. This is because Windows doesn't have a
    #                  notion of POSIX uid/gid/mode.
    #
    #   -R, -r         The -R and -r options are synonymous. Causes directories,
    #                  buckets, and bucket subdirectories to be synchronized
    #                  recursively. If you neglect to use this option gsutil will make
    #                  only the top-level directory in the source and destination URLs
    #                  match, skipping any sub-directories.
    #
    #   -u             When a file/object is present in both the source and
    #                  destination, if mtime is available for both, do not perform
    #                  the copy if the destination mtime is newer.
    #
    #   -U             Skip objects with unsupported object types instead of failing.
    #                  Unsupported object types are Amazon S3 Objects in the GLACIER
    #                  storage class.
    #
    #   -x pattern     Causes files/objects matching pattern to be excluded, i.e., any
    #                  matching files/objects are not copied or deleted. Note that the
    #                  pattern is a `Python regular expression
    #                  <https://docs.python.org/3/howto/regex.html>`_, not a wildcard
    #                  (so, matching any string ending in "abc" would be specified
    #                  using ".*abc$" rather than "*abc"). Note also that the exclude
    #                  path is always relative (similar to Unix rsync or tar exclude
    #                  options). For example, if you run the command:
    #
    #                    gsutil rsync -x "data./.*\.txt$" dir gs://my-bucket
    #
    #                  it skips the file dir/data1/a.txt.
    #
    #                  You can use regex alternation to specify multiple exclusions,
    #                  for example:
    #
    #                    gsutil rsync -x ".*\.txt$|.*\.jpg$" dir gs://my-bucket
    #
    #                  skips all .txt and .jpg files in dir.
    #
    #                  NOTE: When using the Windows cmd.exe command line interpreter,
    #                  use ^ as an escape character instead of \ and escape the |
    #                  character. When using Windows PowerShell, use ' instead of "
    #                  and surround the | character with ".
    options_string = " ".join(options_list) if len(options_list) > 0 else "\b"
    exclude_string = " -x".join(exclude_list) if len(exclude_list) > 0 else "\b"
    command_string = f"gsutil -m rsync -rdu gs://{bucket_name}/{experiment_name} {experiments_root_dir}/"
    log.info(command_string)
    os.system(command_string)


def google_storage_rsync_local_to_gs(
    bucket_name, experiments_root_dir, experiment_name, exclude_list, options_list
):
    # NAME
    #   rsync - Synchronize content of two buckets/directories
    #
    #
    # SYNOPSIS
    #
    #   gsutil rsync [OPTION]... src_url dst_url
    #
    #
    #
    # DESCRIPTION
    #   The gsutil rsync command makes the contents under dst_url the same as the
    #   contents under src_url, by copying any missing files/objects (or those whose
    #   data has changed), and (if the -d option is specified) deleting any extra
    #   files/objects. src_url must specify a directory, bucket, or bucket
    #   subdirectory. For example, to sync the contents of the local directory "data"
    #   to the bucket gs://mybucket/data, you could do:
    #
    #     gsutil rsync data gs://mybucket/data
    #
    #   To recurse into directories use the -r option:
    #
    #     gsutil rsync -r data gs://mybucket/data
    #
    #   If you have a large number of objects to synchronize you might want to use the
    #   gsutil option (see "gsutil help options"), to perform parallel
    #   (multi-threaded/multi-processing) synchronization:
    #
    #     gsutil rsync -r data gs://mybucket/data
    #
    #   The -m option typically will provide a large performance boost if either the
    #   source or destination (or both) is a cloud URL. If both source and
    #   destination are file URLs the -m option will typically thrash the disk and
    #   slow synchronization down.
    #
    #   Note 1: Shells (like bash, zsh) sometimes attempt to expand wildcards in ways
    #   that can be surprising. Also, attempting to copy files whose names contain
    #   wildcard characters can result in problems. For more details about these
    #   issues see the section "POTENTIALLY SURPRISING BEHAVIOR WHEN USING WILDCARDS"
    #   under "gsutil help wildcards".
    #
    #   Note 2: If you are synchronizing a large amount of data between clouds you
    #   might consider setting up a
    #   `Google Compute Engine <https://cloud.google.com/products/compute-engine>`_
    #   account and running gsutil there. Since cross-provider gsutil data transfers
    #   flow through the machine where gsutil is running, doing this can make your
    #   transfer run significantly faster than running gsutil on your local
    #   workstation.
    #
    #   Note 3: rsync does not copy empty directory trees, since Cloud Storage uses a
    #   `flat namespace <https://cloud.google.com/storage/docs/folders>`_.
    #
    #
    # Using -d Option (with caution!) to mirror source and destination.
    #   The rsync -d option is very useful and commonly used, because it provides a
    #   means of making the contents of a destination bucket or directory match those
    #   of a source bucket or directory. This is done by copying all data from the
    #   source to the destination and deleting all other data in the destination that
    #   is not in the source. Please exercise caution when you
    #   use this option: It's possible to delete large amounts of data accidentally
    #   if, for example, you erroneously reverse source and destination.
    #
    #   To make the local directory my-data the same as the contents of
    #   gs://mybucket/data and delete objects in the local directory that are not in
    #   gs://mybucket/data:
    #
    #     gsutil rsync -d -r gs://mybucket/data my-data
    #
    #   To make the contents of gs://mybucket2 the same as gs://mybucket1 and delete
    #   objects in gs://mybucket2 that are not in gs://mybucket1:
    #
    #     gsutil rsync -d -r gs://mybucket1 gs://mybucket2
    #
    # You can also mirror data across local directories. This example will copy all
    # objects from dir1 into dir2 and delete all objects in dir2 which are not in dir1:
    #
    #     gsutil rsync -d -r dir1 dir2
    #
    #   To mirror your content across clouds:
    #
    #     gsutil rsync -d -r gs://my-gs-bucket s3://my-s3-bucket
    #
    #   Change detection works if the other Cloud provider is using md5 or CRC32. AWS
    #   multipart upload has an incompatible checksum.
    #
    #   As mentioned above, using -d can be dangerous because of how quickly data can
    #   be deleted. For example, if you meant to synchronize a local directory from
    #   a bucket in the cloud but instead run the command:
    #
    #     gsutil rsync -r -d ./your-dir gs://your-bucket
    #
    #   and your-dir is currently empty, you will quickly delete all of the objects in
    #   gs://your-bucket.
    #
    #   You can also cause large amounts of data to be lost quickly by specifying a
    #   subdirectory of the destination as the source of an rsync. For example, the
    #   command:
    #
    #     gsutil rsync -r -d gs://your-bucket/data gs://your-bucket
    #
    #   would cause most or all of the objects in gs://your-bucket to be deleted
    #   (some objects may survive if there are any with names that sort lower than
    #   "data" under gs://your-bucket/data).
    #
    #   In addition to paying careful attention to the source and destination you
    #   specify with the rsync command, there are two more safety measures you can
    #   take when using gsutil rsync -d:
    #
    #   1. Try running the command with the rsync -n option first, to see what it
    #      would do without actually performing the operations. For example, if
    #      you run the command:
    #
    #        gsutil rsync -r -d -n gs://your-bucket/data gs://your-bucket
    #
    #      it will be immediately evident that running that command without the -n
    #      option would cause many objects to be deleted.
    #
    #   2. Enable object versioning in your bucket, which allows you to restore
    #      objects if you accidentally delete them. For more details see
    #      `Object Versioning
    #      <https://cloud.google.com/storage/docs/object-versioning>`_.
    #
    #
    # BE CAREFUL WHEN SYNCHRONIZING OVER OS-SPECIFIC FILE TYPES (SYMLINKS, DEVICES,
    # ETC.) Running gsutil rsync over a directory containing operating
    # system-specific file types (symbolic links, device files, sockets, named pipes,
    # etc.) can cause various problems. For example, running a command like:
    #
    #     gsutil rsync -r ./dir gs://my-bucket
    #
    #   will cause gsutil to follow any symbolic links in ./dir, creating objects in
    #   my-bucket containing the data from the files to which the symlinks point. This
    #   can cause various problems:
    #
    #   * If you use gsutil rsync as a simple way to backup a directory to a bucket,
    #     restoring from that bucket will result in files where the symlinks used
    #     to be. At best this is wasteful of space, and at worst it can result in
    #     outdated data or broken applications -- depending on what is consuming
    #     the symlinks.
    #
    #   * If you use gsutil rsync over directories containing broken symlinks,
    #     gsutil rsync will abort (unless you pass the -e option).
    #
    #   * gsutil rsync skips symlinks that point to directories.
    #
    #   Since gsutil rsync is intended to support data operations (like moving a data
    #   set to the cloud for computational processing) and it needs to be compatible
    #   both in the cloud and across common operating systems, there are no plans for
    #   gsutil rsync to support operating system-specific file types like symlinks.
    #
    #   We recommend that users do one of the following:
    #
    #   * Don't use gsutil rsync over directories containing symlinks or other OS-
    #     specific file types.
    #   * Use the -e option to exclude symlinks or the -x option to exclude
    #     OS-specific file types by name.
    #   * Use a tool (such as tar) that preserves symlinks and other OS-specific file
    #     types, packaging up directories containing such files before uploading to
    #     the cloud.
    #
    #
    # EVENTUAL CONSISTENCY WITH NON-GOOGLE CLOUD PROVIDERS
    #   While Google Cloud Storage is strongly consistent, some cloud providers
    #   only support eventual consistency. You may encounter scenarios where rsync
    #   synchronizes using stale listing data when working with these other cloud
    #   providers. For example, if you run rsync immediately after uploading an
    #   object to an eventually consistent cloud provider, the added object may not
    #   yet appear in the provider's listing. Consequently, rsync will miss adding
    #   the object to the destination. If this happens you can rerun the rsync
    #   operation again later (after the object listing has "caught up").
    #
    #
    #
    # CHECKSUM VALIDATION AND FAILURE HANDLING
    #   At the end of every upload or download, the gsutil rsync command validates
    #   that the checksum of the source file/object matches the checksum of the
    #   destination file/object. If the checksums do not match, gsutil deletes
    #   the invalid copy and print a warning message. This very rarely happens, but
    #   if it does, please contact gs-team@google.com.
    #
    #   The rsync command retries failures when it is useful to do so, but if
    #   enough failures happen during a particular copy or delete operation, or if
    #   a failure isn't retryable, the overall command fails.
    #
    #   If the -C option is provided, the command instead skips failing objects and
    #   moves on. At the end of the synchronization run, if any failures were not
    #   successfully retried, the rsync command reports the count of failures and
    #   exits with non-zero status. At this point you can run the rsync command
    #   again, and gsutil attempts any remaining needed copy and/or delete
    #   operations.
    #
    #   For more details about gsutil's retry handling, see `Retry strategy
    #   <https://cloud.google.com/storage/docs/retry-strategy#tools>`_.
    #
    #
    # CHANGE DETECTION ALGORITHM
    #   To determine if a file or object has changed, gsutil rsync first checks
    #   whether the file modification time (mtime) of both the source and destination
    #   is available. If mtime is available at both source and destination, and the
    #   destination mtime is different than the source, or if the source and
    #   destination file size differ, gsutil rsync will update the destination. If the
    #   source is a cloud bucket and the destination is a local file system, and if
    #   mtime is not available for the source, gsutil rsync will use the time created
    #   for the cloud object as a substitute for mtime. Otherwise, if mtime is not
    #   available for either the source or the destination, gsutil rsync will fall
    #   back to using checksums. If the source and destination are both cloud buckets
    #   with checksums available, gsutil rsync will use these hashes instead of mtime.
    #   However, gsutil rsync will still update mtime at the destination if it is not
    #   present. If the source and destination have matching checksums and only the
    #   source has an mtime, gsutil rsync will copy the mtime to the destination. If
    #   neither mtime nor checksums are available, gsutil rsync will resort to
    #   comparing file sizes.
    #
    #   Checksums will not be available when comparing composite Google Cloud Storage
    #   objects with objects at a cloud provider that does not support CRC32C (which
    #   is the only checksum available for composite objects). See 'gsutil help
    #   compose' for details about composite objects.
    #
    #
    # COPYING IN THE CLOUD AND METADATA PRESERVATION
    #   If both the source and destination URL are cloud URLs from the same provider,
    #   gsutil copies data "in the cloud" (i.e., without downloading to and uploading
    #   from the machine where you run gsutil). In addition to the performance and
    #   cost advantages of doing this, copying in the cloud preserves metadata (like
    #   Content-Type and Cache-Control). In contrast, when you download data from the
    #   cloud it ends up in a file, which has no associated metadata, other than file
    #   modification time (mtime). Thus, unless you have some way to hold on to or
    #   re-create that metadata, synchronizing a bucket to a directory in the local
    #   file system will not retain the metadata other than mtime.
    #
    #   Note that by default, the gsutil rsync command does not copy the ACLs of
    #   objects being synchronized and instead will use the default bucket ACL (see
    #   "gsutil help defacl"). You can override this behavior with the -p option (see
    #   OPTIONS below).
    #
    #
    # SLOW CHECKSUMS
    #   If you find that CRC32C checksum computation runs slowly, this is likely
    #   because you don't have a compiled CRC32c on your system. Try running:
    #
    #     gsutil ver -l
    #
    #   If the output contains:
    #
    #     compiled crcmod: False
    #
    #   you are running a Python library for computing CRC32C, which is much slower
    #   than using the compiled code. For information on getting a compiled CRC32C
    #   implementation, see 'gsutil help crc32c'.
    #
    #
    # LIMITATIONS
    #
    #   1. The gsutil rsync command will only allow non-negative file modification
    #      times to be used in its comparisons. This means gsutil rsync will resort to
    #      using checksums for any file with a timestamp before 1970-01-01 UTC.
    #
    #   2. The gsutil rsync command considers only the live object version in
    #      the source and destination buckets when deciding what to copy / delete. If
    #      versioning is enabled in the destination bucket then gsutil rsync's
    #      replacing or deleting objects will end up creating versions, but the
    #      command doesn't try to make any noncurrent versions match in the source
    #      and destination buckets.
    #
    #   3. The gsutil rsync command does not support copying special file types
    #      such as sockets, device files, named pipes, or any other non-standard
    #      files intended to represent an operating system resource. If you run
    #      gsutil rsync on a source directory that includes such files (for example,
    #      copying the root directory on Linux that includes /dev ), you should use
    #      the -x flag to exclude these files. Otherwise, gsutil rsync may fail or
    #      hang.
    #
    #   4. The gsutil rsync command copies changed files in their entirety and does
    #      not employ the
    #      `rsync delta-transfer algorithm <https://rsync.samba.org/tech_report/>`_
    #      to transfer portions of a changed file. This is because Cloud Storage
    #      objects are immutable and no facility exists to read partial object
    #      checksums or perform partial replacements.
    #
    # OPTIONS
    #   -a canned_acl  Sets named canned_acl when uploaded objects created. See
    #                  "gsutil help acls" for further details. Note that rsync will
    #                  decide whether or not to perform a copy based only on object
    #                  size and modification time, not current ACL state. Also see the
    #                  -p option below.
    #
    #   -c             Causes the rsync command to compute and compare checksums
    #                  (instead of comparing mtime) for files if the size of source
    #                  and destination match. This option increases local disk I/O and
    #                  run time if either src_url or dst_url are on the local file
    #                  system.
    #
    #   -C             If an error occurs, continue to attempt to copy the remaining
    #                  files. If errors occurred, gsutil's exit status will be
    #                  non-zero even if this flag is set. This option is implicitly
    #                  set when running "gsutil rsync...".
    #
    #                  NOTE: -C only applies to the actual copying operation. If an
    #                  error occurs while iterating over the files in the local
    #                  directory (e.g., invalid Unicode file name) gsutil will print
    #                  an error message and abort.
    #
    #   -d             Delete extra files under dst_url not found under src_url. By
    #                  default extra files are not deleted.
    #
    #                  NOTE: this option can delete data quickly if you specify the
    #                  wrong source/destination combination. See the help section
    #                  above, "BE CAREFUL WHEN USING -d OPTION!".
    #
    #   -e             Exclude symlinks. When specified, symbolic links will be
    #                  ignored. Note that gsutil does not follow directory symlinks,
    #                  regardless of whether -e is specified.
    #
    # -i             This forces rsync to skip any files which exist on the
    # destination and have a modified time that is newer than the source file. (If an
    # existing destination file has a modification time equal to the source file's,
    # it will be updated if the sizes are different.)
    #
    #   -j <ext,...>   Applies gzip transport encoding to any file upload whose
    #                  extension matches the -j extension list. This is useful when
    #                  uploading files with compressible content (such as .js, .css,
    #                  or .html files) because it saves network bandwidth while
    #                  also leaving the data uncompressed in Google Cloud Storage.
    #
    #                  When you specify the -j option, files being uploaded are
    #                  compressed in-memory and on-the-wire only. Both the local
    #                  files and Cloud Storage objects remain uncompressed. The
    #                  uploaded objects retain the Content-Type and name of the
    #                  original files.
    #
    #                  Note that if you want to use the top-level -m option to
    #                  parallelize copies along with the -j/-J options, your
    #                  performance may be bottlenecked by the
    #                  "max_upload_compression_buffer_size" boto config option,
    #                  which is set to 2 GiB by default. This compression buffer
    #                  size can be changed to a higher limit, e.g.:
    #
    #                    gsutil -o "GSUtil:max_upload_compression_buffer_size=8G" \
    #                      -m rsync -j html,txt /local/source/dir gs://bucket/path
    #
    #   -J             Applies gzip transport encoding to file uploads. This option
    #                  works like the -j option described above, but it applies to
    #                  all uploaded files, regardless of extension.
    #
    #                  CAUTION: If you use this option and some of the source files
    #                  don't compress well (e.g., that's often true of binary data),
    #                  this option may result in longer uploads.
    #
    #   -n             Causes rsync to run in "dry run" mode, i.e., just outputting
    #                  what would be copied or deleted without actually doing any
    #                  copying/deleting.
    #
    #   -p             Causes ACLs to be preserved when objects are copied. Note that
    #                  rsync will decide whether or not to perform a copy based only
    #                  on object size and modification time, not current ACL state.
    #                  Thus, if the source and destination differ in size or
    #                  modification time and you run gsutil rsync -p, the file will be
    #                  copied and ACL preserved. However, if the source and
    #                  destination don't differ in size or checksum but have different
    #                  ACLs, running gsutil rsync -p will have no effect.
    #
    #                  Note that this option has performance and cost implications
    #                  when using the XML API, as it requires separate HTTP calls for
    #                  interacting with ACLs. The performance issue can be mitigated
    #                  to some degree by using gsutil rsync to cause parallel
    #                  synchronization. Also, this option only works if you have OWNER
    #                  access to all of the objects that are copied.
    #
    #                  You can avoid the additional performance and cost of using
    #                  rsync -p if you want all objects in the destination bucket to
    #                  end up with the same ACL by setting a default object ACL on
    #                  that bucket instead of using rsync -p. See 'gsutil help
    #                  defacl'.
    #
    #   -P             Causes POSIX attributes to be preserved when objects are
    #                  copied.  With this feature enabled, gsutil rsync will copy
    #                  fields provided by stat. These are the user ID of the owner,
    #                  the group ID of the owning group, the mode (permissions) of the
    #                  file, and the access/modification time of the file. For
    #                  downloads, these attributes will only be set if the source
    #                  objects were uploaded with this flag enabled.
    #
    #                  On Windows, this flag will only set and restore access time and
    #                  modification time. This is because Windows doesn't have a
    #                  notion of POSIX uid/gid/mode.
    #
    #   -R, -r         The -R and -r options are synonymous. Causes directories,
    #                  buckets, and bucket subdirectories to be synchronized
    #                  recursively. If you neglect to use this option gsutil will make
    #                  only the top-level directory in the source and destination URLs
    #                  match, skipping any sub-directories.
    #
    #   -u             When a file/object is present in both the source and
    #                  destination, if mtime is available for both, do not perform
    #                  the copy if the destination mtime is newer.
    #
    #   -U             Skip objects with unsupported object types instead of failing.
    #                  Unsupported object types are Amazon S3 Objects in the GLACIER
    #                  storage class.
    #
    #   -x pattern     Causes files/objects matching pattern to be excluded, i.e., any
    #                  matching files/objects are not copied or deleted. Note that the
    #                  pattern is a `Python regular expression
    #                  <https://docs.python.org/3/howto/regex.html>`_, not a wildcard
    #                  (so, matching any string ending in "abc" would be specified
    #                  using ".*abc$" rather than "*abc"). Note also that the exclude
    #                  path is always relative (similar to Unix rsync or tar exclude
    #                  options). For example, if you run the command:
    #
    #                    gsutil rsync -x "data./.*\.txt$" dir gs://my-bucket
    #
    #                  it skips the file dir/data1/a.txt.
    #
    #                  You can use regex alternation to specify multiple exclusions,
    #                  for example:
    #
    #                    gsutil rsync -x ".*\.txt$|.*\.jpg$" dir gs://my-bucket
    #
    #                  skips all .txt and .jpg files in dir.
    #
    #                  NOTE: When using the Windows cmd.exe command line interpreter,
    #                  use ^ as an escape character instead of \ and escape the |
    #                  character. When using Windows PowerShell, use ' instead of "
    #                  and surround the | character with ".
    options_string = " ".join(options_list) if len(options_list) > 0 else ""
    exclude_string = " -x".join(exclude_list) if len(exclude_list) > 0 else ""

    command_string = f"gsutil -m rsync -rdu {experiments_root_dir}/{experiment_name} gs://{bucket_name}/"
    log.info(command_string)
    os.system(command_string)


class GoogleStorageClient(object):
    def __init__(self, config: GoogleStorageClientConfig):
        super(GoogleStorageClient, self).__init__()
        self.config = config
        self.directory_structure_cache = None
        self.bucket_name = config.bucket_name
        self.storage_client, self.bucket_client = get_bucket_client(
            bucket_name=self.bucket_name
        )

    def get_remote_dir_tree(self):
        self.bucket_client.list_blobs(
            max_results=None,
            page_token=None,
            prefix=self.config.experiment_name,
            delimiter="/",
            start_offset=None,
            end_offset=None,
            include_trailing_delimiter=None,
            versions=None,
            projection="noAcl",
            fields=None,
            client=None,
        )

    def upload_file(self, filepath_on_bucket, local_filepath):
        upload_file(
            bucket_name=self.bucket_name,
            source_file_name=local_filepath,
            destination_file_name=filepath_on_bucket,
        )

    def download_file(self, filepath_on_bucket, local_filepath):
        download_file(
            bucket_name=self.bucket_name,
            source_file_name=filepath_on_bucket,
            destination_file_name=local_filepath,
        )
