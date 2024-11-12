import sys
from typing import cast
from re import sub

from cv2 import (
    COLOR_RGB2GRAY,
    THRESH_BINARY,
    THRESH_OTSU,
    cvtColor,
    threshold,
)
from numpy import array, ndarray
from PIL import Image
from pytesseract import image_to_string
from torch import (
    Tensor,
    cat,
    float32,
    load,
    multinomial,
    no_grad,
    softmax,
    tensor,
    zeros,
)
from torch import device as Device
from torch.cuda import is_available
from torch.nn import LSTM, Linear, Module
from torch.nn.functional import pad


class HandwritingTextRecognizer(Module):
    def __init__(
        self,
        input_size: int,
        hidden_units: int,
        output_shape: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.rnn = LSTM(
            input_size,
            self.hidden_units,
            self.num_layers,
            batch_first=True,
        )
        self.fc = Linear(
            hidden_units,
            output_shape,
        )

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return (
            zeros(self.num_layers, batch_size, self.hidden_units, device=device),
            zeros(self.num_layers, batch_size, self.hidden_units, device=device),
        )


def extract_text(image_path: str) -> str:
    preprocessed_img = preprocess_image(image_path)
    pil_image = Image.fromarray(preprocessed_img)
    text = image_to_string(pil_image, config="--psm 11")
    return text.strip()


def sample(predictions: Tensor, temperature: float = 1.0) -> int | float | bool:
    predictions = predictions / temperature
    probabilities = softmax(predictions, dim=-1)
    return multinomial(probabilities, 1).item()


def generate_text(
    model: HandwritingTextRecognizer,
    seed_text: str,
    max_length: int,
    device: Device,
    temperature: float = 1.0,
) -> str:
    model.eval()
    generated_text = seed_text

    seed_tensor = (
        tensor([ord(char) for char in seed_text], dtype=float32).unsqueeze(0).to(device)
    )

    if seed_tensor.size(-1) < 16384:
        padding_size = 16384 - seed_tensor.size(-1)
        seed_tensor = pad(seed_tensor, (0, padding_size), "constant", 0)
    else:
        seed_tensor = seed_tensor[:, :16384]

    seed_tensor = seed_tensor.unsqueeze(1)

    hidden = model.init_hidden(seed_tensor.size(0), device)

    print(f"Initial seed_tensor shape: {seed_tensor.shape}")
    print(f"Initial Hidden state shape: {hidden[0].shape}, {hidden[1].shape}")

    with no_grad():
        for _ in range(max_length):
            output, hidden = model(seed_tensor, hidden)

            output = output.squeeze(1)

            predicted_char_index = sample(output, temperature)
            predicted_char = chr(predicted_char_index)
            generated_text += predicted_char

            # Update seed tensor with the new character
            new_char_tensor = (
                tensor([ord(predicted_char)], dtype=float32).unsqueeze(0).to(device)
            )
            new_char_tensor = pad(
                new_char_tensor,
                (0, 16384 - new_char_tensor.size(-1)),
                "constant",
                0,
            )
            new_char_tensor = new_char_tensor.unsqueeze(1)

            # Slide the window to include the new character
            seed_tensor = cat((seed_tensor[:, 1:, :], new_char_tensor), dim=1)

    return generated_text


def preprocess_image(
    image_path: str,
    # target_size: tuple[int, int] = (128, 128),
) -> ndarray:
    img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img_arr = array(img)
    gray_img = cvtColor(img_arr, COLOR_RGB2GRAY)
    _, thresh_image = threshold(gray_img, 0, 255, THRESH_BINARY + THRESH_OTSU)
    # resized_image = resize(thresh_image, target_size)
    return thresh_image


def recognize_text(
    model: HandwritingTextRecognizer,
    image_path: str,
    device: Device,
) -> str:
    return generate_text(model, extract_text(image_path), 200, device)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("Usage: python __main__.py <image_path>")
        return

    image_path = argv[1]
    device = Device("cuda" if is_available() else "cpu")

    model = cast(
        HandwritingTextRecognizer,
        load("handwriting_text_recognizer.pth"),
    ).to(device)

    recognized_text = recognize_text(model, image_path, device)
    recognized_text = sub(r"[^\x32-\x7F\s]", "", recognized_text)
    print(f"Recognized Text: '\x1b[1;31m{recognized_text}\x1b[0m'")


if __name__ == "__main__":
    main(sys.argv)
