import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_sample_image(output_folder):
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255

    cv2.rectangle(image, (50, 80), (250, 250), (255, 0, 0), -1)
    cv2.circle(image, (420, 180), 80, (0, 255, 0), -1)
    cv2.line(image, (50, 330), (550, 330), (0, 0, 255), 5)

    image_file = output_folder / "sample_image.png"
    cv2.imwrite(str(image_file), image)

    return image


def display_image(title, image, is_gray=False):
    plt.figure(figsize=(8, 5))

    if is_gray:
        plt.imshow(image, cmap="gray")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def process_image(image, output_folder):
    original_file = output_folder / "original_image.png"
    grayscale_file = output_folder / "grayscale_image.png"
    resized_file = output_folder / "resized_image.png"
    blurred_file = output_folder / "blurred_image.png"
    edges_file = output_folder / "edges_image.png"
    threshold_file = output_folder / "threshold_image.png"
    detected_file = output_folder / "detected_objects.png"

    cv2.imwrite(str(original_file), image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(grayscale_file), gray)

    resized = cv2.resize(image, (300, 200))
    cv2.imwrite(str(resized_file), resized)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(str(blurred_file), blurred)

    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite(str(edges_file), edges)

    _, threshold = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    cv2.imwrite(str(threshold_file), threshold)

    contours, hierarchy = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected_image = image.copy()

    object_count = 0

    print("Detected Objects")
    print("----------------")

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 500:
            object_count += 1

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                detected_image,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            print(f"Object {object_count}")
            print(f"Area: {area:.2f}")
            print(f"Bounding Box: x={x}, y={y}, width={w}, height={h}")
            print()

    cv2.imwrite(str(detected_file), detected_image)

    return {
        "gray": gray,
        "resized": resized,
        "blurred": blurred,
        "edges": edges,
        "threshold": threshold,
        "detected_image": detected_image,
        "object_count": object_count
    }


def main():
    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)

    image = create_sample_image(output_folder)

    print("Image Information")
    print("-----------------")
    print(f"Image shape: {image.shape}")
    print(f"Image height: {image.shape[0]}")
    print(f"Image width: {image.shape[1]}")
    print(f"Color channels: {image.shape[2]}")

    processed = process_image(image, output_folder)

    print(f"Total detected objects: {processed['object_count']}")

    display_image("Original Image", image)
    display_image("Grayscale Image", processed["gray"], is_gray=True)
    display_image("Edge Detection", processed["edges"], is_gray=True)
    display_image("Threshold Image", processed["threshold"], is_gray=True)
    display_image("Detected Objects", processed["detected_image"])

    print()
    print("Processed images saved in the outputs folder.")


main()
________________________________________


