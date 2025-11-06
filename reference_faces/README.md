# Reference Faces Directory

Place reference face images in this directory for face recognition.

## Instructions

1. Add one image per person
2. Name the file after the person (e.g., `john_doe.jpg`, `jane_smith.png`)
3. The filename (without extension) will be used as the person's name in captions

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

## Example Structure

```
reference_faces/
├── alice.jpg
├── bob.png
├── charlie.jpg
└── diana.jpeg
```

## Tips

- Use clear, frontal face photos
- Good lighting improves recognition accuracy
- One face per image works best
- Higher resolution images (but not too large) work better

## Troubleshooting

If face recognition isn't working:
1. Ensure the reference image has a clearly visible face
2. Try different lighting conditions
3. Check that face-recognition library is installed: `pip install face-recognition`
4. Adjust tolerance in `face_recognition_module.py` (default: 0.6)

