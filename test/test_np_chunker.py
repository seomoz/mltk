
import unittest

from mltk.aptagger import FastPerceptronTagger
from mltk.np_chunker import NPChunker

tagger = FastPerceptronTagger()
chunker = NPChunker()


class TestNPChunker(unittest.TestCase):
    def setUp(self):
        self.text_tags_iob = [
            [('Pierre', 'NNP', 'B'), ('Vinken', 'NNP', 'I'), (',', ',', 'O'),
            ('61', 'CD', 'B'), ('years', 'NNS', 'I'), ('old', 'JJ', 'O'),
            (',', ',', 'O'), ('will', 'MD', 'O'), ('join', 'VB', 'O'),
            ('the', 'DT', 'B'), ('board', 'NN', 'I'), ('as', 'IN', 'O'),
            ('a', 'DT', 'B'), ('nonexecutive', 'JJ', 'I'),
            ('director', 'NN', 'I'), ('Nov.', 'NNP', 'B'),
            ('29', 'CD', 'I'), ('.', '.', 'O')],

            [('Mr.', 'NNP', 'B'), ('Vinken', 'NNP', 'I'), ('is', 'VBZ', 'O'),
            ('chairman', 'NN', 'B'), ('of', 'IN', 'O'),
            ('Elsevier', 'NNP', 'B'), ('N.V.', 'NNP', 'I'),
            (',', ',', 'O'), ('the', 'DT', 'B'), ('Dutch', 'NNP', 'I'),
            ('publishing', 'NN', 'I'), ('group', 'NN', 'I'),
            ('.', '.', 'O')],

            [('Rudolph', 'NNP', 'B'), ('Agnew', 'NNP', 'I'),
            (',', ',', 'O'), ('55', 'CD', 'B'), ('years', 'NNS', 'I'),
            ('old', 'JJ', 'O'), ('and', 'CC', 'O'), ('former', 'JJ', 'O'),
            ('chairman', 'NN', 'B'), ('of', 'IN', 'O'),
            ('Consolidated', 'NNP', 'B'), ('Gold', 'NNP', 'I'),
            ('Fields', 'NNP', 'I'), ('PLC', 'NNP', 'I'), (',', ',', 'O'),
            ('was', 'VBD', 'O'), ('named', 'VBN', 'O'), ('a', 'DT', 'B'),
            ('nonexecutive', 'JJ', 'I'), ('director', 'NN', 'I'),
            ('of', 'IN', 'O'), ('this', 'DT', 'B'), ('British', 'JJ', 'I'),
            ('industrial', 'JJ', 'I'), ('conglomerate', 'NN', 'I'),
            ('.', '.', 'O')]]

    def test_chunk_sents_iob(self):
        '''
        The chunker returns reasonable IOB labels
        '''
        text_tags = [[(t[0], t[1]) for t in sent]
            for sent in self.text_tags_iob]
        iob_labels = chunker.chunk_sents(text_tags, True)
        self.assertListEqual(self.text_tags_iob, iob_labels)

    def test_chunk_sents(self):
        '''
        The chunker returns noun phrases
        '''
        text_tags = [[(t[0], t[1]) for t in sent] 
            for sent in self.text_tags_iob]
        noun_phrases = chunker.chunk_sents(text_tags)
        expected_phrases = [
    [[('Pierre', 'NNP'), ('Vinken', 'NNP')],
    [('61', 'CD'), ('years', 'NNS')],
    [('the', 'DT'), ('board', 'NN')],
    [('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN')],
    [('Nov.', 'NNP'), ('29', 'CD')]],

    [[('Mr.', 'NNP'), ('Vinken', 'NNP')],
    [('chairman', 'NN')],
    [('Elsevier', 'NNP'), ('N.V.', 'NNP')],
    [('the', 'DT'), ('Dutch', 'NNP'), ('publishing', 'NN'), ('group', 'NN')]],

    [[('Rudolph', 'NNP'), ('Agnew', 'NNP')],
    [('55', 'CD'), ('years', 'NNS')],
    [('chairman', 'NN')],
    [('Consolidated', 'NNP'),
    ('Gold', 'NNP'),
    ('Fields', 'NNP'),
    ('PLC', 'NNP')],
    [('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN')],
    [('this', 'DT'),
    ('British', 'JJ'),
    ('industrial', 'JJ'),
    ('conglomerate', 'NN')]]]

        self.assertEqual(noun_phrases, expected_phrases)
    
    def test_empty(self):
        '''
        The chunker works OK if it's passed empty tokens, or empty sentences
        '''
        sentences = [
            [],
            [('The', 'DT'), ('empty', 'NN'), ('', 'NN'), ('POS', ''), ('', ''),
                ('.', '.')]]
        chunks = chunker.chunk_sents(sentences, True)
        self.assertEqual(
            chunks,
            [[], 
            [('The', 'DT', 'B'),
              ('empty', 'NN', 'I'),
              ('', 'NN', 'I'),
              ('POS', '', 'O'),
              ('', '', 'O'),
              ('.', '.', 'O')]]
        )

        np_chunks = chunker.chunk_sents(sentences)
        self.assertEqual(
            np_chunks,
            [[], [[('The', 'DT'), ('empty', 'NN'), ('', 'NN')]]])

    def test_unicode(self):
        sentence = u'Beyonc\u00e9 performed during half time of Super Bowl'
        sentence += ' XLVII .'
        tags = tagger.tag(sentence.encode('utf-8').split())

        # unicode must be decoded to utf-8 or it raises an error
        tags_unicode = [[(token.decode('utf-8'), tag) for token, tag in tags]]
        self.assertRaises(
            UnicodeEncodeError, chunker.chunk_sents, tags_unicode, True)

        # but things work if the tokens are encoded as bytes
        iob_labels = chunker.chunk_sents([tags])[0]
        self.assertEqual(
            iob_labels,
                [[('Beyonc\xc3\xa9', 'NNP')],
                 [('half', 'DT'), ('time', 'NN')],
                 [('Super', 'NNP'), ('Bowl', 'NNP'), ('XLVII', 'NNP')]])


if __name__ == '__main__':
    unittest.main()

