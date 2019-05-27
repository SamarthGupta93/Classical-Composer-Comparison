import constants
from helper import get_datamap
import matplotlib.pyplot as plt
composers = {
	constants.CHOPIN:False,
	constants.BEETHOVEN:True,
	constants.MOZART:False,
	constants.SCHUBERT:False,
	constants.SCHUMANN:True
}
timesteps = 20000
beethoven_start = 10000
schumann_start = 0000
mozart_start = 40000
skip = 32
def get_plotting_data(data_map, composer, timesteps, start=0, skip=4):
	pitch_data = []
	timesteps_data = []
	composer_data = data_map[composer]
	composer_data = composer_data[start:start+timesteps]
	for timestep in range(0, timesteps, skip):
		data = [idx for idx, e in enumerate(composer_data[timestep]) if e==1]
		for value in data:
			pitch_data.append(value)
			timesteps_data.append((timestep)//skip)

	return timesteps_data, pitch_data

data_map = get_datamap(composers, split=False)

if composers[constants.BEETHOVEN]: beethoven_timesteps_data, beethoven_pitch_data = get_plotting_data(data_map, composer=constants.BEETHOVEN, timesteps=timesteps, start=beethoven_start, skip=skip)
if composers[constants.MOZART]: mozart_timesteps_data, mozart_pitch_data = get_plotting_data(data_map, composer=constants.MOZART, timesteps=timesteps, start=mozart_start, skip=skip)
if composers[constants.SCHUMANN]: schumann_timesteps_data, schumann_pitch_data = get_plotting_data(data_map, composer=constants.SCHUMANN, timesteps=timesteps, start=schumann_start, skip=skip)


plt.plot(beethoven_timesteps_data, beethoven_pitch_data)
plt.plot(schumann_timesteps_data, schumann_pitch_data)
#plt.plot(mozart_timesteps_data, mozart_pitch_data)


plt.legend([constants.BEETHOVEN, constants.SCHUMANN])
plt.xlabel('Timesteps')
plt.ylabel('Pitch')
plt.title('Beethoven vs Schumann compositions comparison')

plt.show()