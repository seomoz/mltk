
#include <iostream>
#include <tr1/unordered_map>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <tr1/functional>

#include "../ext/murmur3.c"

// DEBUG
#include <fstream>
#include <sstream>

#include <boost/timer.hpp>


// use a custom hash function for the string... evidently murmurhash
// is pretty fast
uint64_t murmurhash3(const std::string& key)
{
    uint64_t ret[2];   // need 128 bits of space to hold result
    MurmurHash3_x64_128(key.c_str(), key.length(), 5, ret);
    return ret[0];
}


/*
perceptron tagger implemented as follows

Want to predict C classes with F features and a bias
Each feature value is a string, e.g. for the "word suffix" feature
    the value is "ing", etc..

features = N dim array
each entry is hash map of word->C dimensional float

For each feature value combination we have some class weights represented
as a C dimensional float.

Predict sums up the C dim arrays for each feature value
*/

// features used to predict a given POS
typedef std::vector<std::string> features_t;

// weights for individual classes for each feature
typedef std::vector<float> class_weights_t;
typedef class_weights_t bias_weights_t;

// the weights for one feature (word -> class weights)
typedef std::tr1::unordered_map<std::string, class_weights_t,
    std::tr1::function<unsigned long(const std::string&)> > one_weight_t;

// all weights for all features
typedef std::vector<one_weight_t> weights_t;

// some words always have a defined tag
typedef std::tr1::unordered_map<std::string, std::string,
    std::tr1::function<unsigned long(const std::string&)> > tagmap_t;

// tags are tuples of (token, tag)
typedef std::vector<std::pair<std::string, std::string> > tag_t;

// it's easier to pass things in as std::map from cython
// since it doesn't require dragging along the custom hash...
typedef std::vector<std::pair<std::string, float> > class_weights_in_t;
typedef std::vector<std::map<std::string, class_weights_in_t> > weights_in_t;
typedef std::map<std::string, std::string> tagmap_in_t;



//------------------------------------------------------
// utilities
std::string normalize(std::string& word)
{
    // normalize a word.
    // - All words are lower cased
    // - 4 letter digits are represented as !YEAR
    // - Other digits are represented as !DIGITS
    if (word.find("-") != std::string::npos && word != "-")
        return "!HYPHEN";
    else if (word.find_first_not_of("0123456789") == std::string::npos &&
        word.length() == 4)
        return "!YEAR";
    else if (std::isdigit(word[0]))
        return "!DIGITS";
    else
    {
        // lowercase
        std::string ret;
        ret.assign(word);
        std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
        return ret;
    }
}

std::string last_n_letters(std::string& s, int n)
{
    // get the last three letters (or less) of the string
    if (s.length() > n)
        return s.substr(s.length() - n);
    else
        return s.substr(0);
}


const std::string SPACE = " ";

std::string join(std::string& s1, std::string& s2)
{
    // join with a space in the center
    // find the length of the final string, reserve it then append
    std::string sout;
    sout.reserve(s1.length() + s2.length() + 1);
    sout.append(s1);
    sout.append(SPACE);
    sout.append(s2);
    return sout;
}

void get_features(std::size_t k,
    std::string& word,
    std::vector<std::string>& context,
    std::string& prev,
    std::string& prev2,
    features_t& features)
{
    // create some features for the given word
    std::size_t i = k + 2;

    std::string suffix;

    features.clear();

    suffix = last_n_letters(word, 3);
    features.push_back(suffix);
    features.push_back(word.substr(0, 1));
    features.push_back(prev);
    features.push_back(prev2);
    features.push_back(join(prev, prev2));
    features.push_back(context[i]);
    features.push_back(join(prev, context[i]));
    features.push_back(context[i-1]);
    suffix = last_n_letters(context[i-1], 3);
    features.push_back(suffix);
    features.push_back(context[i-2]);
    features.push_back(context[i+1]);
    suffix = last_n_letters(context[i+1], 3);
    features.push_back(suffix);
    features.push_back(context[i+2]);
}


// since the classes are fixed we'll predefine them here
const std::string POS_TAGS[] =
{
    "#", "$", "\'\'", ",", "-LRB-", "-RRB-", ".", ":", "CC", "CD", "DT",
    "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS",
    "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM",
    "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$",
    "WRB", "``"
};
const int NTAGS = sizeof(POS_TAGS) / sizeof(POS_TAGS[0]);
const int NFEATURES = 13;

class AveragedPerceptron
{
    public:
        AveragedPerceptron(weights_in_t weights,
            class_weights_in_t bias_weights);
        ~AveragedPerceptron();
        std::string predict(features_t& features);

    private:
        weights_t weights;
        bias_weights_t bias_weights;

        // disable some default constructors
        AveragedPerceptron();
        AveragedPerceptron& operator= (const AveragedPerceptron& other);
        AveragedPerceptron(const AveragedPerceptron& other);
};

AveragedPerceptron::AveragedPerceptron(
    weights_in_t weights, class_weights_in_t bias_weights) :
    weights(), bias_weights(NTAGS, 0.0)
{
    // a mapping from class name to index
    std::map<std::string, std::size_t> class_map;
    for (std::size_t k = 0; k < NTAGS; ++k)
    {
        class_map[POS_TAGS[k]] = k;
    }

    // populate the bias weight vector
    for (class_weights_in_t::iterator it = bias_weights.begin();
        it != bias_weights.end(); ++it)
        this->bias_weights[class_map[it->first]] = it->second;

    // now the weight vectors
    for (weights_in_t::iterator it = weights.begin(); it != weights.end(); ++it)
    {
        one_weight_t one_weight(100, murmurhash3);
        one_weight.max_load_factor(0.1);   // to avoid collisons

        // it iterates over a map->vector(pair)
        std::map<std::string, class_weights_in_t>::iterator itw;
        for (itw = it->begin(); itw != it->end(); ++itw)
        {
            // itw->first = the word
            // itw->second is vector of pair we'll turn to dense array
            class_weights_t feature_vec(NTAGS, 0.0);
            for (class_weights_in_t::iterator itc = itw->second.begin();
                    itc != itw->second.end(); ++itc)
                feature_vec[class_map[itc->first]] = itc->second;
            one_weight[itw->first] = feature_vec;
        }
        this->weights.push_back(one_weight);
    }
}

AveragedPerceptron::~AveragedPerceptron() {}

std::string AveragedPerceptron::predict(features_t& features)
{
    // make a prediction - add all the class scores from the features/weights
    // and return the max

    // initialize to the bias weights
    std::vector<float> scores(bias_weights);

    // now read through the features
    one_weight_t::iterator got;
    for (std::size_t k = 0; k < NFEATURES; ++k)
    {
        std::string feature = features[k];
        got = weights[k].find(feature);
        if (got != weights[k].end())
        {
            // this feature exists.  update the scores
            for (std::size_t i = 0; i < NTAGS; ++i)
                scores[i] += got->second[i];
        }
    }

    // now find the maximum class associated with the scores
    float max_score = -1.0e-20;
    std::string chosen_class;
    for (std::size_t i = 0; i < NTAGS; ++i)
    {
        if (scores[i] > max_score)
        {
            max_score = scores[i];
            chosen_class = POS_TAGS[i];
        }
    }

    return chosen_class;
}


class PerceptronTagger
{
    public:
        PerceptronTagger(weights_in_t weights, class_weights_in_t bias_weights,
            tagmap_in_t specified_tags);
        ~PerceptronTagger();

        // tags a single sentence
        tag_t tag_sentence(std::vector<std::string>& sentence);

        // tag a document that has been sentence and word tokenized
        void tag_sentences(
            std::vector<std::vector<std::string> >& document,
            std::vector<tag_t>& tags
        );

    private:
        tagmap_t specified_tags;
        AveragedPerceptron model;

        // disable some default constructors
        PerceptronTagger();
        PerceptronTagger& operator= (const PerceptronTagger& other);
        PerceptronTagger(const PerceptronTagger& other);

};

PerceptronTagger::PerceptronTagger(
    weights_in_t weights, class_weights_in_t bias_weights,
    tagmap_in_t specified_tags) :
    specified_tags(20000, murmurhash3), model(weights, bias_weights)
{
    this->specified_tags.insert(specified_tags.begin(), specified_tags.end());
}

PerceptronTagger::~PerceptronTagger() {}

tag_t PerceptronTagger::tag_sentence(std::vector<std::string>& sentence)
{
    // tag a single sentence
    tag_t tags;
    tags.clear();

    // make the context for each word
    std::vector<std::string> context;
    context.push_back("-START-"); context.push_back("-START2-");
    for (std::vector<std::string>::iterator it = sentence.begin();
            it != sentence.end(); ++it)
        context.push_back(normalize(*it));
    context.push_back("-END-"); context.push_back("-END2-");

    // now tag each word
    std::string prev = "-START-";
    std::string prev2 = "-START2-";

    features_t features;
    tagmap_t::iterator got;

    std::string word;
    std::string tag;

    for (std::size_t i=0; i < sentence.size(); ++i)
    {
        word = sentence[i];

        // check if the word is in the set of specified tags
        got = specified_tags.find(word);
        if (got != specified_tags.end())
        {
            tag = got->second;
        }
        else
        {
            // we'll make some features and run the model to predict
            // the results
            get_features(i, word, context, prev, prev2, features);
            tag = model.predict(features);
        }
        tags.push_back(std::make_pair<std::string, std::string>(word, tag));

        prev2 = prev;
        prev = tag;
    }

    return tags;
}

void PerceptronTagger::tag_sentences(
    std::vector<std::vector<std::string> >& document,
    std::vector<tag_t>& tags)
{
    tags.clear();
    std::vector<std::vector<std::string> >::iterator it;
    for (it = document.begin(); it != document.end(); ++it)
        tags.push_back(tag_sentence(*it));
}

