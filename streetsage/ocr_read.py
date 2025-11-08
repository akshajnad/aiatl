"""
OCR module for reading text on demand.
Uses pytesseract for fast, local OCR.
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OCRReader:
    """OCR wrapper for reading signs and text."""

    def __init__(self):
        """Initialize OCR reader."""
        # Test if pytesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            logger.warning("OCR features will be disabled")

    def preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess image region for better OCR.

        Args:
            roi: Image region (BGR)

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Resize if too small (OCR works better on larger text)
        h, w = gray.shape
        if h < 100 or w < 100:
            scale = max(100 / h, 100 / w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)

        # Adaptive threshold for better contrast
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def read_text(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Read text from frame or ROI.

        Args:
            frame: Input image (BGR)
            roi: Optional (x1, y1, x2, y2) region to read from

        Returns:
            Extracted text (cleaned)
        """
        try:
            # Extract ROI if specified
            if roi is not None:
                x1, y1, x2, y2 = roi
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                image = frame[y1:y2, x1:x2]
            else:
                image = frame

            # Preprocess
            preprocessed = self.preprocess_for_ocr(image)

            # Run OCR
            text = pytesseract.image_to_string(preprocessed, config='--psm 6')

            # Clean text
            text = text.strip()
            text = ' '.join(text.split())  # Normalize whitespace

            return text

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""

    def read_center_text(self, frame: np.ndarray, roi_ratio: float = 0.5) -> str:
        """
        Read text from center region of frame.

        Args:
            frame: Input image (BGR)
            roi_ratio: Fraction of frame to use as ROI

        Returns:
            Extracted text
        """
        h, w = frame.shape[:2]
        margin_x = int(w * (1 - roi_ratio) / 2)
        margin_y = int(h * (1 - roi_ratio) / 2)

        roi = (margin_x, margin_y, w - margin_x, h - margin_y)
        return self.read_text(frame, roi)

    def read_top_text(self, frame: np.ndarray, top_ratio: float = 0.3) -> str:
        """
        Read text from top portion of frame (for signs).

        Args:
            frame: Input image (BGR)
            top_ratio: Fraction of frame height to read from top

        Returns:
            Extracted text
        """
        h, w = frame.shape[:2]
        top_h = int(h * top_ratio)
        roi = (0, 0, w, top_h)
        return self.read_text(frame, roi)

    def get_first_n_words(self, text: str, n: int = 12) -> str:
        """
        Get first N words from text.

        Args:
            text: Input text
            n: Number of words

        Returns:
            First N words
        """
        words = text.split()
        return ' '.join(words[:n])

    def detect_text_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential text regions using MSER.

        Args:
            frame: Input image (BGR)

        Returns:
            List of (x1, y1, x2, y2) regions
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # MSER (Maximally Stable Extremal Regions) for text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)

            # Get bounding boxes
            bboxes = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                # Filter by aspect ratio and size
                aspect = w / max(h, 1)
                if 0.1 < aspect < 10 and w > 20 and h > 10:
                    bboxes.append((x, y, x + w, y + h))

            return bboxes

        except Exception as e:
            logger.error(f"Text region detection error: {e}")
            return []


def read_sign_on_demand(frame: np.ndarray, reader: Optional[OCRReader] = None) -> str:
    """
    Utility function to read sign from current frame.

    Args:
        frame: Input image
        reader: Optional OCRReader instance

    Returns:
        First 8-12 words from sign
    """
    if reader is None:
        reader = OCRReader()

    # Try top portion first (most signs are at top)
    text = reader.read_top_text(frame, top_ratio=0.4)

    # If nothing found, try center
    if not text:
        text = reader.read_center_text(frame, roi_ratio=0.6)

    # Return first 12 words
    return reader.get_first_n_words(text, n=12)


if __name__ == "__main__":
    # Test with webcam
    logging.basicConfig(level=logging.INFO)

    reader = OCRReader()
    cap = cv2.VideoCapture(0)

    print("Press 's' to read sign, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("OCR Test - Press 's' to read", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("Reading...")
            text = read_sign_on_demand(frame, reader)
            print(f"Text: {text}")

    cap.release()
    cv2.destroyAllWindows()
