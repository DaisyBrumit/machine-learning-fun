import pandas as pd
import pomegranate as pg

# Main function
def main():
    states = ['fair', 'loaded'] # Q where N = 2\n",
    obs = ['3', '5', '4', '2', '1', '6', '5', '4', '6', '6'] # This is O where T = 10, V = 1,2,3,4,5,6 where v = 6\n",

    init_prob = {'fair': 0.5, 'loaded': 0.5} # PI where N = 2\n",

    # emission probabilities = B = bi(ot) = P(observation 't' resulting from state 'i')\n",
    emission_prob = {
        'fair': [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        'loaded': [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]
       }

    # transition probabilities = A = a11, ..., aij, ..., aNN = P(moving from state 'i' to state 'j')\n",
    trans_prob = {
        'fair': {'fair': 0.95, 'loaded': 0.095}, # P('fair' state given either previous state)\n",
        'loaded': {'fair': 0.045, 'loaded': 0.9}, # P('loaded' state given either previous state)\n",
        'end': {'fair': 0.005, 'loaded': 0.05} # P(termination given either previous state)\n",
        }

    model = pg.HiddenMarkovModel(name='Model')
    fair_emission = pg.UniformDistribution(1, 6)
    fair_state = pg.State(fair_emission, name='fair')
    loaded_emission = pg.DiscreteDistribution({1: 1/10, 2: 1/10, 3: 1/10, 4: 1/10, 5: 1/10, 6: 1/2})
    loaded_state = pg.State(loaded_emission, name='loaded')

    model.add_states(fair_state, loaded_state)
    model.add_transition(model.start, fair_state, 0.5)
    model.add_transition(model.start, loaded_state, 0.5)

    model.add_transition(fair_state, fair_state, 0.95)
    model.add_transition(fair_state, loaded_state, 0.045)
    model.add_transition(fair_state, model.end, 0.005)

    model.add_transition(loaded_state, fair_state, 0.095)
    model.add_transition(loaded_state, loaded_state, 0.9)
    model.add_transition(loaded_state, model.end, 0.005)

    model.bake()

    col_order = ['Model-start', 'fair', 'loaded', 'Model-end']
    col_names = [s.name for s in model.states]
    index_order = [col_names.index(c) for c in col_order]
    transitions = model.dense_transition_matrix()[:, index_order][index_order,:]
    #print(transitions)

    obs = [3, 5, 4, 2, 1, 6, 5, 4, 6, 6]
    fwd_bkwd = model.predict(obs, algorithm='map') # map = maximum a posteriori = forward/backward algorithm
    fwd_bkwd_prob = model.predict_proba(obs)

    print('observation: ', obs, '\nhmm prediction: ', fwd_bkwd, '\nhmm state 0: fair\nhmm state 1: loaded')

    # training set
    seqs = [[1,3,5,2,3,6,1,1,3,5,4,2,1,6,5,4,6,6,3,6,4,6,4,6,2,3,5,4,5,4,3,1,5,2,3,6,1,6,2,5,6,6,3,6,6,2,4,1,2,5,6,1,5,4,3,6,1,5,6,6,6,1,2,6,3,4,1,6,2,5,1,2,6,4,6,6,1,2,5,1,2,6,5,1,4,2,5,3,4,5,1,6,4,6,4,3,5,1,6,6,4,6,5,1,2,3,5,4,1,4,6,1,2,5,4,1,2,4,1,4,6,3,6,3,6],
            [6,5,4,5,6,3,6,4,6,4,6,2,1,3,5,2,3,6,4,1,2,5,4,2,1,6,1,6,2,5,5,6,3,6,6,3,5,4,5,4,3,1,5,2,3,2,4,1,2,5,6,1,5,4,3,6,4,6,6,5,6,1,1,5,3,6,6,1,2,6,3,2,6,4,6,6,1,2,5,1,2,4,5,6,4,2,3,1,4,5,1,6,4,6,4,3,5,1,6,6,4,6,5,1,2,6,5,4,1,4,6,4,6,3,6,3,6,1,2,5,4,1,2,6,6],
            [4,6,3,6,3,6,1,3,5,2,3,6,6,1,2,5,4,2,1,6,5,4,5,6,3,6,4,6,4,6,2,3,5,4,5,4,3,1,5,2,3,6,1,6,2,5,5,6,3,6,6,2,4,1,2,5,6,1,5,4,3,6,1,4,3,6,6,1,2,6,3,4,6,6,5,6,1,6,1,2,2,6,4,6,5,1,2,6,5,1,4,2,5,3,3,5,1,6,4,6,4,3,6,4,6,5,1,6,6,1,2,6,5,4,1,4,6,1,2,5,4,1,2,6,6],
            [1,5,3,6,6,1,6,6,6,2,6,4,4,1,5,4,3,6,4,5,3,5,2,1,6,6,6,1,6,2,5,5,6,3,6,6,3,5,4,5,4,3,1,5,2,3,2,4,1,2,2,4,1,6,3,6,6,6,6,1,2,1,3,5,2,3,1,6,1,2,5,4,2,1,2,6,1,2,6,4,6,4,3,5,1,6,6,5,1,2,3,5,1,4,2,5,2,5,1,5,1,2,6,5,4,1,4,6,1,2,5,4,1,2,6,6,4,6,5,4,6,3,6,3,6]]

    labels = [['fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded'],
              ['loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair'],
              ['loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair'],
              ['loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','fair','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded','loaded']]

    print('\n\nMODEL FIT SUMMARY:\n', model.fit(sequences=seqs, algorithm='viterbi'))

if __name__ == '__main__':
    main()



