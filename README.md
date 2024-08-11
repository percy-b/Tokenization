# Tokenization
Byte Pair Encoding Tokenization (BPE)

# usage
1. prepare corpus/text data with relevant special tokens if need be
2. from Tokenizer import BPETokenizer, as in the Test notebook, if you have the classes in your notebook or python file
   there is no need to import.
   While importing, ensure the folder Tokenizer is in ur current working directory
2. call the tokenizer class with input parameters, text, vocab_size and allowed_special characters 
    e.g `tokenizer = MyTokenizer(text, vocab_size=500, allowed_special=["|<endoftext>|", "|<eos>|"])`
3. train the tokenizer
    e.g `tokenizer.train()`
4. then use it to encode any new text to get tokens
    e.g `ids = tokenizer.encode("Hello world")`
5. you can finally do any decoding
    e.g `tokenizer.decode(ids)`
6. to get vocabulary, u can use the get_vocabulary function
   e.g `tokenizer.get_vocabulary()`