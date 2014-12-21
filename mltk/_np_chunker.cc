
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <functional>

#include "_utils.cc"


/// features used to predict a given IOB label
typedef std::vector<std::string> np_features_t;

/// the feature weights
typedef std::vector<float> np_weights_t;

/// some words always have a defined label
typedef std::unordered_map<std::string, char,
    std::function<unsigned long(const std::string&)> > np_labelmap_t;

/** it's easier to pass things in as std::map from cython
 since it doesn't require dragging along the custom hash... */
typedef std::map<std::string, char> np_labelmap_in_t;

/// Input to the chunker are already tokenized, POS tagged sentences
typedef std::pair<std::string, std::string> tag_t;

struct iob_t {
    std::string token;
    std::string tag;
    char label;
    iob_t(std::string token, std::string tag, char label) :
        token(token), tag(tag), label(label) {}
};

/// Output from the chunker are tags + the IOB labels
typedef std::vector<iob_t> iob_label_t;


void get_features(std::size_t k,
    std::string const & word,
    std::vector<std::string>& context,
    std::vector<std::string>& tags,
    np_features_t& features)
{
    /** Create some features for the current position.
    word = the current word, un-normalized
    context = the current sentence, normalized, padded w/ -START-/-END-
    tags = the POS tags for the sentence, padded
    features = the return 
    */
    features.clear();
    features.reserve(20);

    std::size_t i = k + 2;

    // unigram words
    features.push_back(join("w-2", 3, context[i-2]));
    features.push_back(join("w-1", 3, context[i-1]));
    features.push_back(join("w0", 2, context[i]));
    features.push_back(join("w1", 2, context[i+1]));
    features.push_back(join("w2", 2, context[i+2]));

    // bigram words
    features.push_back(join("w-1w0", 5, context[i-1], context[i]));
    features.push_back(join("w0w1", 4, context[i], context[i+1]));

    // unigram tags
    features.push_back(join("t-2", 3, tags[i-2]));
    features.push_back(join("t-1", 3, tags[i-1]));
    features.push_back(join("t0", 2, tags[i]));
    features.push_back(join("t1", 2, tags[i+1]));
    features.push_back(join("t2", 2, tags[i+2]));

    // bigram tags
    features.push_back(join("t-2t-1", 6, tags[i-2], tags[i-1]));
    features.push_back(join("t-1t0", 5, tags[i-1], tags[i]));
    features.push_back(join("t0t1", 4, tags[i], tags[i+1]));
    features.push_back(join("t1t2", 4, tags[i+1], tags[i+2]));

    // trigram tags
    features.push_back(join("t-2t-1t0", 8, tags[i-2], tags[i-1], tags[i]));
    features.push_back(join("t-1t0t1", 7, tags[i-1], tags[i], tags[i+1]));
    features.push_back(join("t0t1t2", 6, tags[i], tags[i+1], tags[i+2]));

    // first letter
    features.push_back(join('p', word[0]));
}


// the dimension of our hashed feature vector (2 ** 17)
#define N_FEATURES 131072
// we'll use bit wise & instead of mod in the feature hash...
#define N_FEATURES_MINUS_1 131071
// the bias weights start at this index (N_FEATURES * N_CLASSES)
#define BIAS_INDEX 393216


inline uint64_t feature_hash(const std::string& key)
{
    /// The hash function for our feature hashing
    return murmurhash3(key) & N_FEATURES_MINUS_1;
}

// the number of possible output classes (I, O, B)
#define N_CLASSES 3

class FastNPChunker : public TaggerBase<tag_t, iob_t>
{
    public:
        FastNPChunker(np_weights_t weights, np_labelmap_in_t labelmap_in);
        ~FastNPChunker();

        /// Given a POS tagged sentence, return IOB labels for each token
        iob_label_t tag_sentence(std::vector<tag_t> const & sentence);

    private:
        // the weights are logically a 2D matrix of (n_features, n_classes)
        // but are stored as a flattened array running across rows
        // then down columns.  Thus the weights for feature k are
        // in entries (k * N_CLASSES):(k * N_CLASSES + N_CLASSES)
        np_weights_t weights;

        // if the word is in labelmap then it always has a predefined label
        np_labelmap_t labelmap;

        // the output class labels
        std::vector<char> classes;

        /// Given some features, compute the scores for each class
        void compute_scores(np_features_t const & features,
            std::vector<float>& scores);

        // disable some default constructors
        FastNPChunker();
        FastNPChunker& operator= (const FastNPChunker& other);
        FastNPChunker(const FastNPChunker& other);
};

FastNPChunker::FastNPChunker(
    np_weights_t weights, np_labelmap_in_t labelmap_in) :
    weights(weights), labelmap(1000, murmurhash3), classes()
{
    // fill in the labelmap
    for (np_labelmap_in_t::const_iterator it = labelmap_in.begin();
        it != labelmap_in.end(); ++it)
    {
        labelmap[it->first] = it->second;
    }

    // fill in the classes
    classes.reserve(3);
    classes.push_back('I');
    classes.push_back('O');
    classes.push_back('B');
}

FastNPChunker::~FastNPChunker() {}

void FastNPChunker::compute_scores(np_features_t const & features,
    std::vector<float>& scores)
{
    // process:
    // 1.  initialize the scores to the bias weights
    // 2.  for each feature, compute its hash, add in the weights

    // 1.  the bias weights are the last N_CLASSES entries in the weight
    //  vector
    for (std::size_t k = 0; k < N_CLASSES; ++k)
        scores[k] = weights[BIAS_INDEX + k];

    // 2.
    for (np_features_t::const_iterator it = features.begin();
        it != features.end(); ++ it)
    {
        // this is the starting index for these feature weights
        uint64_t index = feature_hash(*it) * N_CLASSES;
        for (std::size_t k = 0; k < N_CLASSES; ++k)
            scores[k] += weights[index + k];
    }
}

iob_label_t FastNPChunker::tag_sentence(std::vector<tag_t> const & sentence)
{
    iob_label_t ret;
    ret.reserve(sentence.size());

    // make the word and tag context
    std::vector<std::string> context;
    std::vector<std::string> tag_context;
    context.reserve(sentence.size() + 4);
    tag_context.reserve(sentence.size() + 4);
    context.push_back("-START-"); context.push_back("-START2-");
    tag_context.push_back("-START-"); tag_context.push_back("-START2-");
    for (std::vector<tag_t>::const_iterator it = sentence.begin();
        it != sentence.end(); ++it)
    {
        context.push_back(normalize(it->first));
        tag_context.push_back(it->second);
    }
    context.push_back("-END-"); context.push_back("-END2-");
    tag_context.push_back("-END-"); tag_context.push_back("-END2-");

    // loop through the sentence and assign class to each token
    np_features_t features;
    std::vector<float> scores(N_CLASSES, 0.0);

    // need to keep track of last label to check for invalid sequences
    // Since 'I' is not a valid label for the first word of the sentence,
    // and OI is the only invalid sequence this handles the first word
    // of the sentence case
    char last_label = 'O';

    for (std::size_t i=0; i < sentence.size(); ++i)
    {
        char label;
        std::string const & word = sentence[i].first;

        // check if word is in the labelmap
        np_labelmap_t::const_iterator got = labelmap.find(word);
        if (got != labelmap.end())
        {
            label = got->second;
        }
        else
        {
            get_features(i, word, context, tag_context, features);
            compute_scores(features, scores);

            // scores holds the class predictions
            // predicted class is the maximum value in scores, except for
            // invalid orderings ('O' then 'I')
            float max_score = -1e20;
            for (std::size_t k=0; k < N_CLASSES; ++k)
            {
                if (scores[k] > max_score &&
                        // not an invalid combo
                        !(last_label == 'O' && classes[k] == 'I'))
                {
                    max_score = scores[k];
                    label = classes[k];
                }
            }
        }

        last_label = label;
        ret.push_back(iob_t(word, sentence[i].second, label));
    }

    return ret;
}


