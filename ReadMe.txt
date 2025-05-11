Cheeky!!!

Using OpenCV and Haar Cascade classifier, this code detects the face of the person and find the regions which, supposedly, its cheeks should be. Then, it proceeds to calculate the mean RGB color of those regions. Why, you might be asking? The idea is to detect the person's skin color!

Works only with one face per image (we filter multiple faces to get only the most prominent one). Also, use only JPG or PNG images.

Don't forget to install the requirements!

pip install -r requirements.txt

Accepted parameters:

--path: Path to the image. If a directory is sent, it checks all jpgs and pngs from there
--savePath: Path where to save the cheeky image. If empty, it just dont save it
--fileName: Name of the file to save the results. If empty, default is results.txt

Usage example:
python Cheeky.py --path many_images/ --savePath save/
python Cheeky.py --path pretty.jpg --savePath save/ --fileName so_pretty.txt

That's all folks!