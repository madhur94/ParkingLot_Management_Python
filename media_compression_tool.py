import heapq
import os
import pickle
from collections import defaultdict
import cv2
import numpy as np

class Node:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_table(data):
    freq = defaultdict(int)
    for byte in data:
        freq[byte] += 1
    return freq

def build_huffman_tree(freq_table):
    heap = [Node(symbol=k, freq=v) for k, v in freq_table.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(symbol=None, freq=node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

def build_codes(root):
    codes = {}
    def _build_codes_helper(node, current_code=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = current_code
            return
        _build_codes_helper(node.left, current_code + "0")
        _build_codes_helper(node.right, current_code + "1")
    _build_codes_helper(root)
    return codes

def compress_data(data, codes):
    return ''.join(codes[byte] for byte in data)

def pad_data(encoded_data):
    extra_bits = 8 - len(encoded_data) % 8
    encoded_data += "0" * extra_bits
    padded_info = "{0:08b}".format(extra_bits)
    return padded_info + encoded_data

def to_byte_array(padded_data):
    return bytearray(int(padded_data[i:i+8], 2) for i in range(0, len(padded_data), 8))

def compress_file(input_path, output_path):
    with open(input_path, "rb") as f:
        data = f.read()

    freq_table = build_frequency_table(data)
    huffman_tree = build_huffman_tree(freq_table)
    codes = build_codes(huffman_tree)

    encoded_data = compress_data(data, codes)
    padded_data = pad_data(encoded_data)
    byte_data = to_byte_array(padded_data)

    with open(output_path, "wb") as out:
        out.write(byte_data)

    with open(output_path + ".meta", "wb") as meta:
        pickle.dump((codes, len(encoded_data)), meta)

    print(f"[HUFFMAN] Compressed {input_path} -> {output_path}")
    return os.path.getsize(input_path), os.path.getsize(output_path)

def decompress_file(input_path, output_path):
    with open(input_path, "rb") as f:
        byte_data = f.read()

    with open(input_path + ".meta", "rb") as meta:
        codes, original_length = pickle.load(meta)

    reverse_codes = {v: k for k, v in codes.items()}
    bitstring = ''.join(f"{byte:08b}" for byte in byte_data)
    extra_padding = int(bitstring[:8], 2)
    bitstring = bitstring[8:-extra_padding]

    current_code = ""
    decoded_bytes = bytearray()

    for bit in bitstring:
        current_code += bit
        if current_code in reverse_codes:
            decoded_bytes.append(reverse_codes[current_code])
            current_code = ""

    with open(output_path, "wb") as out:
        out.write(decoded_bytes)

    print(f"[HUFFMAN] Decompressed {input_path} -> {output_path}")

def get_decoded_filename(original_path):
    base, ext = os.path.splitext(original_path)
    return f"{base}_decoded{ext}"

def dct_compress_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    resized = cv2.resize(image, (width//2, height//2))
    restored = cv2.resize(resized, (width, height))
    cv2.imwrite(output_path, restored)
    print(f"[DCT] Compressed {image_path} -> {output_path}")
    return os.path.getsize(image_path), os.path.getsize(output_path)

def dct_compress_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width // 2, height // 2))
        restored = cv2.resize(resized, (width, height))
        out.write(restored)

    cap.release()
    out.release()
    print(f"[DCT] Compressed {video_path} -> {output_path}")
    return os.path.getsize(video_path), os.path.getsize(output_path)

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    image_path = "images/parking_lot_1.png"
    video_path = "videos/parking_lot_1.mp4"

    # Huffman
    huff_img_out = image_path + ".huff"
    huff_vid_out = video_path + ".huff"
    orig_img_size, huff_img_size = compress_file(image_path, huff_img_out)
    orig_vid_size, huff_vid_size = compress_file(video_path, huff_vid_out)

    # Decompress for Huffman
    huff_decoded_img = get_decoded_filename(image_path)
    huff_decoded_vid = get_decoded_filename(video_path)
    decompress_file(huff_img_out, huff_decoded_img)
    decompress_file(huff_vid_out, huff_decoded_vid)

    # DCT
    dct_img_out = "images/parking_lot_1_dct.png"
    dct_vid_out = "videos/parking_lot_1_dct.mp4"
    orig_img_size_dct, dct_img_size = dct_compress_image(image_path, dct_img_out)
    orig_vid_size_dct, dct_vid_size = dct_compress_video(video_path, dct_vid_out)

    # Print sizes
    print("\nFILE SIZE COMPARISON:")
    print(f"[IMAGE] Original: {orig_img_size} bytes | Huffman: {huff_img_size} bytes | DCT: {dct_img_size} bytes")
    print(f"[VIDEO] Original: {orig_vid_size} bytes | Huffman: {huff_vid_size} bytes | DCT: {dct_vid_size} bytes")

    os.system(f"python main.py --image images/parking_lot_1_decoded.png --data data/coordinates_1.yml --video videos/parking_lot_1_decoded.mp4 --start-frame 400")

