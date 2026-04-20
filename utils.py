import socket
import pickle
import struct

def send_msg(sock, data):
    """Sends a pickled object with a 4-byte length prefix."""
    try:
        payload = pickle.dumps(data)
        length = struct.pack('>Q', len(payload))
        sock.sendall(length + payload)
    except (BrokenPipeError, ConnectionResetError):
        print("Connection lost.")
    except Exception as e:
        print(f"Error sending message: {e}")


def recv_msg(sock):
    """Receives a pickled object with a 4-byte length prefix."""
    try:
        raw_len = recvall(sock, 8)
        if not raw_len:
            return None
        length = struct.unpack('>Q', raw_len)[0]
        payload = recvall(sock, length)
        if not payload:
            return None
        return pickle.loads(payload)
    except (ConnectionResetError, EOFError):
        print("Connection closed by the other end.")
        return None
    except Exception as e:
        print(f"Error receiving message: {e}")
        return None

def recvall(sock, n):
    """Helper function to receive n bytes or return None if EOF is hit."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
