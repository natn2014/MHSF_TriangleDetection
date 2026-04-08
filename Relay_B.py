#!/usr/bin/env python3
"""Waveshare Modbus POE ethernet relay board. See https://www.waveshare.com/wiki/Modbus_POE_ETH_Relay.
See docs for changing the IP address & configuration settings.
"""
import socket               
import time
import threading

class Relay():
    """Waveshare Modbus POE ethernet relay board."""
    def __init__(self, host='192.168.1.200', port=502, address=0x01):
        
        self.host = host
        self.port = port
        self.address = address
        self.channels = range(1, 9)

        # Table of CRC values for high–order byte
        self.CRCTableHigh = [
          0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
          0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
          0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01,
          0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
          0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81,
          0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
          0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01,
          0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
          0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
          0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
          0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01,
          0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
          0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
          0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
          0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01,
          0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
          0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
          0x40
        ]

        # Table of CRC values for low–order byte
        self.CRCTableLow = [
          0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06, 0x07, 0xC7, 0x05, 0xC5, 0xC4,
          0x04, 0xCC, 0x0C, 0x0D, 0xCD, 0x0F, 0xCF, 0xCE, 0x0E, 0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09,
          0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9, 0x1B, 0xDB, 0xDA, 0x1A, 0x1E, 0xDE, 0xDF, 0x1F, 0xDD,
          0x1D, 0x1C, 0xDC, 0x14, 0xD4, 0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
          0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3, 0xF2, 0x32, 0x36, 0xF6, 0xF7,
          0x37, 0xF5, 0x35, 0x34, 0xF4, 0x3C, 0xFC, 0xFD, 0x3D, 0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A,
          0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38, 0x28, 0xE8, 0xE9, 0x29, 0xEB, 0x2B, 0x2A, 0xEA, 0xEE,
          0x2E, 0x2F, 0xEF, 0x2D, 0xED, 0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
          0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60, 0x61, 0xA1, 0x63, 0xA3, 0xA2,
          0x62, 0x66, 0xA6, 0xA7, 0x67, 0xA5, 0x65, 0x64, 0xA4, 0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F,
          0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB, 0x69, 0xA9, 0xA8, 0x68, 0x78, 0xB8, 0xB9, 0x79, 0xBB,
          0x7B, 0x7A, 0xBA, 0xBE, 0x7E, 0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
          0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71, 0x70, 0xB0, 0x50, 0x90, 0x91,
          0x51, 0x93, 0x53, 0x52, 0x92, 0x96, 0x56, 0x57, 0x97, 0x55, 0x95, 0x94, 0x54, 0x9C, 0x5C,
          0x5D, 0x9D, 0x5F, 0x9F, 0x9E, 0x5E, 0x5A, 0x9A, 0x9B, 0x5B, 0x99, 0x59, 0x58, 0x98, 0x88,
          0x48, 0x49, 0x89, 0x4B, 0x8B, 0x8A, 0x4A, 0x4E, 0x8E, 0x8F, 0x4F, 0x8D, 0x4D, 0x4C, 0x8C,
          0x44, 0x84, 0x85, 0x45, 0x87, 0x47, 0x46, 0x86, 0x82, 0x42, 0x43, 0x83, 0x41, 0x81, 0x80,
          0x40
        ]
         

    def connect(self):
        """Connect to the device."""
        # create socket
        self.sock = socket.socket()
        # set timeout to prevent blocking
        self.sock.settimeout(5.0)  # 5 second timeout
        # connect to device
        self.sock.connect((self.host, self.port))

    def disconnect(self):
        """Disconnect from the device."""
        self.sock.close()

    def ModbusCRC(self, data):
        """Calculate modbus CRC value."""
        crcHigh = 0xff;
        crcLow = 0xff; 
        index = 0;
        for byte in data:
            index = crcLow ^ byte
            crcLow  = crcHigh ^ self.CRCTableHigh[index]
            crcHigh = self.CRCTableLow[index]
        
        return (crcHigh << 8 | crcLow)

    def __str__(self):
        return f'Waveshare Relay host [{self.host}] port [{self.port}] address '
        f'[{self.address}]'

    def _write(self, cmd):
        crc = self.ModbusCRC(cmd)
        cmd.append(crc & 0xFF)
        cmd.append(crc >> 8)
        try:
            self.sock.send(bytearray(cmd))
        except AttributeError as err:
            raise RuntimeError(f'Tried sending a message to {self} but it is '
                'not connected.') from err

    def on(self, channel: int):
        """Turn a relay channel on.

        Args:
            channel: channel number.
        """
        if channel not in self.channels:
            raise ValueError(f'Provided channel [{channel}] not in [{self.channels}] for device [{self}].')

        # driver channels are 0 indexed
        cmd = [self.address, 0x05, 0, channel - 1, 0xFF, 0]     
        cmd_bytes = bytearray(cmd)  # Store original 6 bytes before CRC
        self._write(cmd)

        # Receive response once (8 bytes expected)
        response = self.sock.recv(8)
        
        if len(response) < 8:
            raise RuntimeError(f'Invalid response length from device [{self}] when turning on channel [{channel}]: expected 8, got {len(response)}.')

        # Verify response matches command (first 6 bytes, excludes CRC)
        if response[:6] != cmd_bytes:
            raise RuntimeError(f'Did not receive correct response from device [{self}] when turning on channel [{channel}].')

    def off(self, channel: int):
        """Turn a relay channel off.

        Args:
            channel: channel number.
        """
        if channel not in self.channels:
            raise ValueError(f'Provided channel [{channel}] not in [{self.channels}] for device [{self}].')

        # driver channels are 0 indexed
        cmd = [self.address, 0x05, 0, channel - 1, 0, 0]     
        cmd_bytes = bytearray(cmd)  # Store original 6 bytes before CRC
        self._write(cmd)

        # Receive response once (8 bytes expected)
        response = self.sock.recv(8)
        
        if len(response) < 8:
            raise RuntimeError(f'Invalid response length from device [{self}] when turning off channel [{channel}]: expected 8, got {len(response)}.')

        # Verify response matches command (first 6 bytes, excludes CRC)
        if response[:6] != cmd_bytes:
            raise RuntimeError(f'Did not receive correct response from device [{self}] when turning off channel [{channel}].')

    def status(self, channel: int):
        """Return whether a relay channel is on (True) or off (False).

        Args:
            channel: channel number.
        """
        if channel not in self.channels:
            raise ValueError(f'Provided channel [{channel}] not in [{self.channels}] for device [{self}].')

        cmd = [0x01, 0x01, 0, 0, 0, 0x08, 0x3D, 0xCC]   
        self.sock.send(bytearray(cmd))

        # Receive response (6 bytes expected for read coil status)
        response = self.sock.recv(6)
        
        if len(response) < 6:
            raise RuntimeError(f'Invalid response length when checking status of channel [{channel}]: expected 6, got {len(response)}.')

        # Byte 3 contains the coil status bits
        if response[3] & (1 << (channel - 1)):
            return True
        else:
            return False

    def __enter__(self):
        self.connect()
        if not self.sock:
            raise RuntimeError(f'Failed to connect to device [{self}].')
        print(f'Connected to device [{self}].')
        return self


    def __exit__(self, *args):
        self.disconnect()

    """Turn on the first relay channel."""
    def turn_on_first_relay(self):
        """Turn on the first relay channel."""
        #cmd = [self.address, 0x05, 0, 0, 0xFF, 0]
        #self._write(cmd)
        #self.on(1)

        cmd = [0x01, 0x05 ,0 ,0, 0xFF, 0 ]
        self._write(cmd)
        #print(f"Command sent: {cmd}")
        print(f"Expected response: {bytearray(cmd)}")
        # Check if the response matches the command sent
        print(f"Response received: {self.sock.recv(8)}")
        # If the response does not match, raise an error
        # Note: The response length should be 8 bytes for a valid Modbus TCP response
        if self.sock.recv(8) != bytearray(cmd):
            raise RuntimeError(f'Did not receive response from device [{self}] when turning on first relay channel.')

        if not self.status(1):
            raise RuntimeError(f'Failed to turn on first relay channel of device [{self}].')
        
    def check_DI(self):
        """Check the status of digital inputs (DI1 to DI8)."""
        #print("Checking DI status...")
        cmd = [0x01, 0x02, 0x00, 0x00, 0x00, 0x08]
        self._write(cmd)
        #print(f"Command sent: {cmd}")
        response = self.sock.recv(6)
        #print(f"Expected response: {bytearray(cmd)}, received: {response}")

        if len(response) != 6:
            raise RuntimeError(f'Invalid response length from device [{self}]. Expected 6 bytes, got {len(response)} bytes.')

        di_status = response[3]
        for i in range(8):
            if di_status & (1 << i):
                print(f"DI{i+1} is ON")
        return di_status
    
    def DI_on_Relay(self, channel: int):
        """Turn on the relay channel if the corresponding DI is ON.

        Args:
            channel: channel number.
        """
        if channel not in self.channels:
            raise ValueError(f'Provided channel [{channel}] not in [{self.channels}] for device [{self}].')

        di_status = self.check_DI()
        if di_status & (1 << (channel - 1)):
            self.on(channel)
            print(f"Relay channel {channel} turned ON because DI{channel} is ON.")
        else:
            print(f"Relay channel {channel} remains OFF because DI{channel} is OFF.")
            self.off(channel)

    def is_DI_on(self, di_number: int) -> bool:
        """Return True if the specified DI number (1-8) is ON, else False."""
        if di_number < 1 or di_number > 8:
            raise ValueError("DI number must be between 1 and 8")

        # send command and receive status as usual
        cmd = [0x01, 0x02, 0x00, 0x00, 0x00, 0x08]
        self._write(cmd)
        response = self.sock.recv(6)

        if len(response) != 6:
            raise RuntimeError(f"Invalid response length: expected 6, got {len(response)}")

        di_status = response[3]

        # Check if bit for di_number is set
        return bool(di_status & (1 << (di_number - 1)))

    def all_on(self):
        """Turn on all relay channels (1-8)."""
        for channel in self.channels:
            try:
                self.on(channel)
            except Exception as e:
                print(f"Failed to turn on channel {channel}: {e}")

    def all_off(self):
        """Turn off all relay channels (1-8)."""
        for channel in self.channels:
            try:
                self.off(channel)
            except Exception as e:
                print(f"Failed to turn off channel {channel}: {e}")


    def check_DI_periodic(self):
        while True:
            self.check_DI()
            self.DI_on_Relay(1)
            time.sleep(0.5)
            

    def start_check_DI_thread(self):
        thread = threading.Thread(target=self.check_DI_periodic, daemon=True)
        thread.start()
        print("Started periodic DI check thread.")
      
if __name__ == '__main__':
    with Relay(host='192.168.1.201') as relay:
        relay.all_on()
        time.sleep(0.5)
        relay.all_off()
        #time.sleep(0.5)
        #relay.start_check_DI_thread()
        #time.sleep(5)  # Allow some time for periodic DI checks
        #relay.check_DI_periodic()
