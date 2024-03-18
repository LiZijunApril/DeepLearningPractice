# %%
from torch import nn

class Encoder(nn.Module):
    """基本的编码器接口"""
    def __init__(self, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        
    def forward(self, X, *args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    """基本的解码器接口"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """编码器解码器的基本类"""
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
    
    