import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Construction des données

titanic = pd.read_csv('titanic.csv')
titanic['Age'] = titanic['Age'].astype(int)

femmes = titanic[titanic.Sex == 'female']  
hommes = titanic[titanic.Sex == 'male']

femmesSurvie = femmes[femmes.Survived == 1]
hommesSurvie = hommes[hommes.Survived == 1]

genders = ['Femmes','Hommes']
genderStats = [femmes.size,hommes.size]
genderSurviveStats = [femmesSurvie.size,hommesSurvie.size]

avgTicketDead = titanic[titanic.Survived == 0].Fare.mean()
avgTicketAlive = titanic[titanic.Survived == 1].Fare.mean()

aliveState = ['Survivant','Victime']
aliveStateValue = [avgTicketAlive,avgTicketDead]

# Affichages des graphiques

plt.close()

fig = plt.figure(1, figsize=(9, 9))
fig.subplots_adjust(top=0.9)

p1 = plt.subplot(221)
p1.bar(genders, genderStats)
p1.set_title('Proportion Femme/homme')

p2 = plt.subplot(222)
p2.bar(genders, genderSurviveStats)
p2.set_title('Survivants femme/homme')

p3 = plt.subplot(223)
p3.bar(aliveState, aliveStateValue)
p3.set_title('Différence prix survivant/mort (en £1000)')


# Affichage des graphiques

plt.tight_layout()
plt.show()
