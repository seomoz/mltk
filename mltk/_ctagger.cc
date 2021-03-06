
#include <iostream>
#include <unordered_map>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <functional>

#include "_utils.cc"


/**
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

/// features used to predict a given POS
typedef std::vector<std::string> features_t;

/// weights for individual classes for each feature
typedef std::vector<float> class_weights_t;
typedef class_weights_t bias_weights_t;

/// the weights for one feature (word -> class weights)
typedef std::unordered_map<std::string, class_weights_t,
    std::function<unsigned long(const std::string&)> > one_weight_t;

/// all weights for all features
typedef std::vector<one_weight_t> weights_t;

/// some words always have a defined tag
typedef std::unordered_map<std::string, std::string,
    std::function<unsigned long(const std::string&)> > tagmap_t;

/// tags are tuples of (token, tag)
typedef std::pair<std::string, std::string> tag_t;

/** it's easier to pass things in as std::map from cython
 since it doesn't require dragging along the custom hash... */
typedef std::vector<std::pair<std::string, float> > class_weights_in_t;
typedef std::vector<std::map<std::string, class_weights_in_t> > weights_in_t;
typedef std::map<std::string, std::string> tagmap_in_t;



std::string last_n_letters(std::string const & s, int n)
{
    ///< get the last three letters (or less) of the string
    if (s.length() > n)
        return s.substr(s.length() - n);
    else
        return s.substr(0);
}


void get_features(std::size_t k,
    std::string const & word,
    std::vector<std::string>& context,
    std::string& prev,
    std::string& prev2,
    features_t& features)
{
    //< create some features for the given word
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
        std::string predict(features_t const & features);

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

std::string AveragedPerceptron::predict(features_t const & features)
{
    // make a prediction - add all the class scores from the features/weights
    // and return the max

    // initialize to the bias weights
    std::vector<float> scores(bias_weights);

    // now read through the features
    one_weight_t::const_iterator got;
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
    float max_score = -1.0e20;
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

class PerceptronTagger : public TaggerBase<std::string, tag_t>
{
    public:
        PerceptronTagger(weights_in_t weights, class_weights_in_t bias_weights,
            tagmap_in_t specified_tags);
        ~PerceptronTagger();

        /// tags a single sentence
        std::vector<tag_t> tag_sentence(
            std::vector<std::string> const & sentence);

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

std::vector<tag_t> PerceptronTagger::tag_sentence(
    std::vector<std::string> const & sentence)
{
    // tag a single sentence
    std::vector<tag_t> tags;

    // make the context for each word
    std::vector<std::string> context;
    context.reserve(sentence.size() + 4);
    context.push_back("-START-"); context.push_back("-START2-");
    for (std::vector<std::string>::const_iterator it = sentence.begin();
            it != sentence.end(); ++it)
        context.push_back(normalize(*it));
    context.push_back("-END-"); context.push_back("-END2-");

    // now tag each word
    std::string prev = "-START-";
    std::string prev2 = "-START2-";

    features_t features;

    for (std::size_t i=0; i < sentence.size(); ++i)
    {
        std::string tag;
        std::string const & word = sentence[i];

        // check if the word is in the set of specified tags
        tagmap_t::const_iterator got = specified_tags.find(word);
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
        tags.push_back(std::make_pair(word, tag));

        prev2 = prev;
        prev = tag;
    }

    return tags;
}

