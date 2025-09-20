import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DromcomDemographyAnalyzer:
    def __init__(self, territoire_name):
        self.territoire = territoire_name
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#5CAB7D', '#2A9D8F', '#E76F51', '#264653']
        
        self.start_year = 2002
        self.end_year = 2025
        
        # Configuration spécifique à chaque territoire
        self.config = self._get_territoire_config()
        
    def _get_territoire_config(self):
        """Retourne la configuration spécifique pour chaque DROM-COM"""
        configs = {
            "Guadeloupe": {
                "population_base": 390000,
                "natalite_base": 12.5,
                "mortalite_base": 7.2,
                "idh_base": 0.82,
                "specialites": ["tourisme", "agriculture", "services"]
            },
            "Martinique": {
                "population_base": 375000,
                "natalite_base": 11.8,
                "mortalite_base": 7.5,
                "idh_base": 0.84,
                "specialites": ["tourisme", "banane", "rhum", "services"]
            },
            "Guyane": {
                "population_base": 290000,
                "natalite_base": 25.4,
                "mortalite_base": 4.8,
                "idh_base": 0.76,
                "specialites": ["spatial", "or", "biodiversite", "foret"]
            },
            "La Réunion": {
                "population_base": 860000,
                "natalite_base": 15.2,
                "mortalite_base": 6.3,
                "idh_base": 0.80,
                "specialites": ["tourisme", "canne", "services", "numerique"]
            },
            "Mayotte": {
                "population_base": 280000,
                "natalite_base": 35.7,
                "mortalite_base": 4.2,
                "idh_base": 0.69,
                "specialites": ["agriculture", "peche", "jeunesse"]
            },
            "Saint-Martin": {
                "population_base": 35000,
                "natalite_base": 14.3,
                "mortalite_base": 5.8,
                "idh_base": 0.78,
                "specialites": ["tourisme", "commerce", "plages"]
            },
            "Saint-Barthélemy": {
                "population_base": 9800,
                "natalite_base": 9.8,
                "mortalite_base": 6.2,
                "idh_base": 0.88,
                "specialites": ["luxe", "tourisme", "plages"]
            },
            "Saint-Pierre-et-Miquelon": {
                "population_base": 6000,
                "natalite_base": 8.5,
                "mortalite_base": 9.1,
                "idh_base": 0.83,
                "specialites": ["peche", "tourisme", "froid"]
            },
            "Wallis-et-Futuna": {
                "population_base": 11500,
                "natalite_base": 16.2,
                "mortalite_base": 5.4,
                "idh_base": 0.79,
                "specialites": ["traditions", "peche", "agriculture"]
            },
            "Polynésie française": {
                "population_base": 280000,
                "natalite_base": 14.8,
                "mortalite_base": 5.6,
                "idh_base": 0.81,
                "specialites": ["tourisme", "perliculture", "peche"]
            },
            "Nouvelle-Calédonie": {
                "population_base": 270000,
                "natalite_base": 15.3,
                "mortalite_base": 5.9,
                "idh_base": 0.83,
                "specialites": ["nickel", "tourisme", "biodiversite"]
            },
            # Configuration par défaut
            "default": {
                "population_base": 100000,
                "natalite_base": 12.0,
                "mortalite_base": 7.0,
                "idh_base": 0.75,
                "specialites": ["services", "tourisme"]
            }
        }
        
        return configs.get(self.territoire, configs["default"])
    
    def generate_demographic_data(self):
        """Génère des données démographiques pour le territoire"""
        print(f"🏝️ Génération des données démographiques pour {self.territoire}...")
        
        # Créer une base de données annuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Annee': [date.year for date in dates]}
        
        # Données démographiques de base
        data['Population'] = self._simulate_population(dates)
        data['Naissances'] = self._simulate_births(dates)
        data['Deces'] = self._simulate_deaths(dates)
        
        # Taux démographiques (pour 1000 habitants)
        data['Taux_Natalite'] = self._simulate_birth_rate(dates)
        data['Taux_Mortalite'] = self._simulate_death_rate(dates)
        data['Solde_Naturel'] = self._simulate_natural_balance(dates)
        
        # Indice de développement humain
        data['IDH'] = self._simulate_hdi(dates)
        
        # Espérance de vie
        data['Esperance_Vie'] = self._simulate_life_expectancy(dates)
        
        # Migration
        data['Solde_Migratoire'] = self._simulate_migration_balance(dates)
        
        # Structure par âge
        data['Part_Moins_20_Ans'] = self._simulate_young_population(dates)
        data['Part_Plus_60_Ans'] = self._simulate_elderly_population(dates)
        
        # Indicateurs socio-économiques
        data['Taux_Chomage'] = self._simulate_unemployment(dates)
        data['PIB_Par_Habitant'] = self._simulate_gdp_per_capita(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances spécifiques au territoire
        self._add_territory_trends(df)
        
        return df
    
    def _simulate_population(self, dates):
        """Simule la population du territoire"""
        base_population = self.config["population_base"]
        
        population = []
        for i, date in enumerate(dates):
            # Croissance démographique variable selon le territoire
            if self.territoire == "Mayotte":
                growth_rate = 0.035  # Croissance très forte à Mayotte
            elif self.territoire == "Guyane":
                growth_rate = 0.028  # Croissance forte en Guyane
            elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                growth_rate = 0.018  # Croissance modérée dans les petites îles
            elif self.territoire == "Saint-Pierre-et-Miquelon":
                growth_rate = -0.003  # Décroissance à Saint-Pierre-et-Miquelon
            else:
                growth_rate = 0.012  # Croissance modérée ailleurs
                
            growth = 1 + growth_rate * i
            population.append(base_population * growth)
        
        return population
    
    def _simulate_births(self, dates):
        """Simule le nombre de naissances"""
        base_births = self.config["population_base"] * (self.config["natalite_base"] / 1000)
        
        births = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution différente selon les territoires
            if self.territoire == "Mayotte":
                trend = 1 - 0.003 * i  # Légère baisse à Mayotte
            elif self.territoire == "Guyane":
                trend = 1 - 0.002 * i  # Légère baisse en Guyane
            elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                trend = 1 - 0.005 * i  # Baisse plus marquée
            else:
                trend = 1 - 0.004 * i  # Baisse modérée
            
            noise = np.random.normal(1, 0.07)
            births.append(base_births * trend * noise)
        
        return births
    
    def _simulate_deaths(self, dates):
        """Simule le nombre de décès"""
        base_deaths = self.config["population_base"] * (self.config["mortalite_base"] / 1000)
        
        deaths = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution différente selon les territoires (vieillissement)
            if self.territoire in ["Martinique", "Guadeloupe"]:
                trend = 1 + 0.008 * i  # Augmentation due au vieillissement
            elif self.territoire == "Saint-Pierre-et-Miquelon":
                trend = 1 + 0.01 * i  # Forte augmentation
            elif self.territoire in ["Mayotte", "Guyane"]:
                trend = 1 + 0.004 * i  # Faible augmentation (population jeune)
            else:
                trend = 1 + 0.006 * i  # Augmentation modérée
            
            noise = np.random.normal(1, 0.05)
            deaths.append(base_deaths * trend * noise)
        
        return deaths
    
    def _simulate_birth_rate(self, dates):
        """Simule le taux de natalité (pour 1000 habitants)"""
        base_rate = self.config["natalite_base"]
        
        rates = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution différente selon les territoires
            if self.territoire == "Mayotte":
                trend = 1 - 0.015 * i  # Baisse rapide à Mayotte
            elif self.territoire == "Guyane":
                trend = 1 - 0.01 * i  # Baisse modérée en Guyane
            elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                trend = 1 - 0.012 * i  # Baisse marquée
            else:
                trend = 1 - 0.008 * i  # Baisse modérée
            
            noise = np.random.normal(1, 0.04)
            rates.append(base_rate * trend * noise)
        
        return rates
    
    def _simulate_death_rate(self, dates):
        """Simule le taux de mortalité (pour 1000 habitants)"""
        base_rate = self.config["mortalite_base"]
        
        rates = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution différente selon les territoires
            if self.territoire in ["Martinique", "Guadeloupe"]:
                trend = 1 + 0.006 * i  # Augmentation due au vieillissement
            elif self.territoire == "Saint-Pierre-et-Miquelon":
                trend = 1 + 0.008 * i  # Forte augmentation
            elif self.territoire in ["Mayotte", "Guyane"]:
                trend = 1 + 0.003 * i  # Faible augmentation
            else:
                trend = 1 + 0.005 * i  # Augmentation modérée
            
            noise = np.random.normal(1, 0.03)
            rates.append(base_rate * trend * noise)
        
        return rates
    
    def _simulate_natural_balance(self, dates):
        """Simule le solde naturel (naissances - décès)"""
        balance = []
        for i, date in enumerate(dates):
            # Calcul basé sur les naissances et décès simulés
            balance.append(self._simulate_births([date])[0] - self._simulate_deaths([date])[0])
        
        return balance
    
    def _simulate_hdi(self, dates):
        """Simule l'Indice de Développement Humain"""
        base_hdi = self.config["idh_base"]
        
        hdi_values = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Amélioration générale de l'IDH avec des variations selon les territoires
            if self.territoire in ["Saint-Barthélemy", "Martinique"]:
                improvement = 1 + 0.004 * i  # Amélioration lente (déjà élevé)
            elif self.territoire in ["Mayotte", "Guyane"]:
                improvement = 1 + 0.008 * i  # Amélioration plus rapide
            else:
                improvement = 1 + 0.006 * i  # Amélioration modérée
            
            # Ne pas dépasser 0.95 (plafond réaliste)
            new_hdi = min(base_hdi * improvement, 0.95)
            
            noise = np.random.normal(1, 0.01)
            hdi_values.append(new_hdi * noise)
        
        return hdi_values
    
    def _simulate_life_expectancy(self, dates):
        """Simule l'espérance de vie"""
        # Espérance de vie de base selon le territoire
        if self.territoire in ["Martinique", "Guadeloupe", "La Réunion"]:
            base_expectancy = 78.5
        elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
            base_expectancy = 79.2
        elif self.territoire == "Mayotte":
            base_expectancy = 75.8
        elif self.territoire == "Guyane":
            base_expectancy = 76.3
        elif self.territoire == "Saint-Pierre-et-Miquelon":
            base_expectancy = 77.6
        else:
            base_expectancy = 77.0
        
        expectancy = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Amélioration générale de l'espérance de vie
            improvement = 1 + 0.002 * i
            
            # Ne pas dépasser 85 ans (plafond réaliste)
            new_expectancy = min(base_expectancy * improvement, 85)
            
            noise = np.random.normal(1, 0.005)
            expectancy.append(new_expectancy * noise)
        
        return expectancy
    
    def _simulate_migration_balance(self, dates):
        """Simule le solde migratoire"""
        balance = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Solde migratoire variable selon les territoires
            if self.territoire in ["Mayotte", "Guyane"]:
                base_balance = 2000  # Solde positif important
                trend = 1 - 0.02 * i  # Diminution progressive
            elif self.territoire in ["Saint-Barthélemy", "Saint-Martin"]:
                base_balance = 500  # Solde positif modéré
                trend = 1 - 0.01 * i  # Légère diminution
            elif self.territoire == "Saint-Pierre-et-Miquelon":
                base_balance = -100  # Solde négatif
                trend = 1 - 0.005 * i  # Légère amélioration
            else:
                base_balance = 800  # Solde positif modéré
                trend = 1 - 0.015 * i  # Diminution progressive
            
            noise = np.random.normal(1, 0.2)
            balance.append(base_balance * trend * noise)
        
        return balance
    
    def _simulate_young_population(self, dates):
        """Simule la part des moins de 20 ans"""
        # Part de base selon le territoire
        if self.territoire == "Mayotte":
            base_part = 0.55  # Très jeune population
        elif self.territoire == "Guyane":
            base_part = 0.45  # Jeune population
        elif self.territoire in ["Martinique", "Guadeloupe"]:
            base_part = 0.28  # Population vieillissante
        elif self.territoire == "Saint-Pierre-et-Miquelon":
            base_part = 0.22  # Population âgée
        else:
            base_part = 0.32  # Situation intermédiaire
        
        parts = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution différente selon les territoires
            if self.territoire in ["Mayotte", "Guyane"]:
                trend = 1 - 0.008 * i  # Légère baisse (transition démographique)
            elif self.territoire in ["Martinique", "Guadeloupe"]:
                trend = 1 - 0.01 * i  # Baisse plus marquée
            else:
                trend = 1 - 0.009 * i  # Baisse modérée
            
            noise = np.random.normal(1, 0.02)
            parts.append(base_part * trend * noise)
        
        return parts
    
    def _simulate_elderly_population(self, dates):
        """Simule la part des plus de 60 ans"""
        # Part de base selon le territoire
        if self.territoire in ["Martinique", "Guadeloupe"]:
            base_part = 0.25  # Population vieillissante
        elif self.territoire == "Saint-Pierre-et-Miquelon":
            base_part = 0.28  # Population âgée
        elif self.territoire in ["Mayotte", "Guyane"]:
            base_part = 0.08  # Population jeune
        else:
            base_part = 0.18  # Situation intermédiaire
        
        parts = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution différente selon les territoires
            if self.territoire in ["Martinique", "Guadeloupe"]:
                trend = 1 + 0.012 * i  # Augmentation rapide
            elif self.territoire == "Saint-Pierre-et-Miquelon":
                trend = 1 + 0.015 * i  # Augmentation très rapide
            elif self.territoire in ["Mayotte", "Guyane"]:
                trend = 1 + 0.01 * i  # Augmentation modérée
            else:
                trend = 1 + 0.011 * i  # Augmentation modérée
            
            noise = np.random.normal(1, 0.02)
            parts.append(base_part * trend * noise)
        
        return parts
    
    def _simulate_unemployment(self, dates):
        """Simule le taux de chômage"""
        # Taux de base selon le territoire
        if self.territoire in ["Mayotte", "Guyane"]:
            base_rate = 0.22  # Chômage élevé
        elif self.territoire in ["Martinique", "Guadeloupe"]:
            base_rate = 0.18  # Chômage important
        elif self.territoire == "Saint-Barthélemy":
            base_rate = 0.08  # Faible chômage
        else:
            base_rate = 0.12  # Situation intermédiaire
        
        rates = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Évolution avec des variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 1.15
            elif year in [2006, 2012, 2017, 2023]:  # Périodes plus favorables
                multiplier = 0.92
            else:
                multiplier = 1.0
            
            # Tendances à long terme
            if self.territoire in ["Mayotte", "Guyane"]:
                trend = 1 - 0.005 * i  # Légère amélioration
            elif self.territoire in ["Martinique", "Guadeloupe"]:
                trend = 1 - 0.004 * i  # Légère amélioration
            else:
                trend = 1 - 0.003 * i  # Très légère amélioration
            
            noise = np.random.normal(1, 0.05)
            rates.append(base_rate * trend * multiplier * noise)
        
        return rates
    
    def _simulate_gdp_per_capita(self, dates):
        """Simule le PIB par habitant (en milliers d'euros)"""
        # PIB de base selon le territoire
        if self.territoire == "Saint-Barthélemy":
            base_gdp = 35.0  # PIB très élevé
        elif self.territoire == "Nouvelle-Calédonie":
            base_gdp = 28.5  # PIB élevé (nickel)
        elif self.territoire in ["Martinique", "Guadeloupe"]:
            base_gdp = 22.0  # PIB moyen-élevé
        elif self.territoire == "Mayotte":
            base_gdp = 8.5  # PIB faible
        elif self.territoire == "Guyane":
            base_gdp = 15.5  # PIB moyen-faible
        else:
            base_gdp = 18.0  # PIB moyen
        
        gdp_values = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance différente selon les territoires
            if self.territoire == "Nouvelle-Calédonie":
                growth = 1 + 0.018 * i  # Croissance soutenue
            elif self.territoire in ["Mayotte", "Guyane"]:
                growth = 1 + 0.022 * i  # Croissance forte
            elif self.territoire in ["Martinique", "Guadeloupe"]:
                growth = 1 + 0.012 * i  # Croissance modérée
            else:
                growth = 1 + 0.015 * i  # Croissance modérée
            
            # Variations cycliques
            if year in [2008, 2009, 2020, 2021]:  # Crises économiques
                multiplier = 0.95
            elif year in [2006, 2012, 2017, 2023]:  # Périodes fastes
                multiplier = 1.06
            else:
                multiplier = 1.0
            
            noise = np.random.normal(1, 0.04)
            gdp_values.append(base_gdp * growth * multiplier * noise)
        
        return gdp_values
    
    def _add_territory_trends(self, df):
        """Ajoute des tendances réalistes adaptées à chaque territoire"""
        for i, row in df.iterrows():
            year = row['Annee']
            
            # Événements communs à tous les territoires
            if 2008 <= year <= 2009:  # Crise financière mondiale
                df.loc[i, 'Taux_Chomage'] *= 1.12
                df.loc[i, 'PIB_Par_Habitant'] *= 0.96
            
            if 2020 <= year <= 2021:  # Pandémie COVID-19
                df.loc[i, 'Taux_Mortalite'] *= 1.08
                df.loc[i, 'PIB_Par_Habitant'] *= 0.92
                df.loc[i, 'Taux_Chomage'] *= 1.15
            
            # Événements spécifiques à certains territoires
            if self.territoire == "Mayotte":
                if year >= 2011:  # Départementalisation
                    df.loc[i, 'IDH'] *= 1.01
                    df.loc[i, 'PIB_Par_Habitant'] *= 1.02
            
            if self.territoire == "Guyane":
                if year in [2017, 2018]:  # Mouvements sociaux
                    df.loc[i, 'PIB_Par_Habitant'] *= 0.97
                    df.loc[i, 'Taux_Chomage'] *= 1.08
            
            if self.territoire == "Nouvelle-Calédonie":
                if year in [2018, 2020, 2021]:  # Référendums et incertitudes politiques
                    df.loc[i, 'Solde_Migratoire'] *= 0.8
                    df.loc[i, 'PIB_Par_Habitant'] *= 0.98
            
            if self.territoire == "La Réunion":
                if year >= 2010:  # Développement du numérique
                    df.loc[i, 'IDH'] *= 1.005
                    df.loc[i, 'PIB_Par_Habitant'] *= 1.01
    
    def create_demographic_analysis(self, df):
        """Crée une analyse complète des indicateurs démographiques"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Évolution de la population
        ax1 = plt.subplot(4, 2, 1)
        self._plot_population_evolution(df, ax1)
        
        # 2. Natalité et mortalité
        ax2 = plt.subplot(4, 2, 2)
        self._plot_birth_death_rates(df, ax2)
        
        # 3. Structure par âge
        ax3 = plt.subplot(4, 2, 3)
        self._plot_age_structure(df, ax3)
        
        # 4. Indice de développement humain
        ax4 = plt.subplot(4, 2, 4)
        self._plot_hdi_evolution(df, ax4)
        
        # 5. Solde naturel et migratoire
        ax5 = plt.subplot(4, 2, 5)
        self._plot_balances(df, ax5)
        
        # 6. Espérance de vie
        ax6 = plt.subplot(4, 2, 6)
        self._plot_life_expectancy(df, ax6)
        
        # 7. Indicateurs économiques
        ax7 = plt.subplot(4, 2, 7)
        self._plot_economic_indicators(df, ax7)
        
        # 8. Projection démographique
        ax8 = plt.subplot(4, 2, 8)
        self._plot_demographic_projection(df, ax8)
        
        plt.suptitle(f'Analyse Démographique de {self.territoire} - DROM-COM ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.territoire}_demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Générer les insights
        self._generate_demographic_insights(df)
    
    def _plot_population_evolution(self, df, ax):
        """Plot de l'évolution de la population"""
        ax.plot(df['Annee'], df['Population'], label='Population', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        
        ax.set_title('Évolution de la Population', fontsize=12, fontweight='bold')
        ax.set_ylabel('Population')
        ax.grid(True, alpha=0.3)
        
        # Ajouter le taux de croissance en second axe
        ax2 = ax.twinx()
        growth_rates = df['Population'].pct_change() * 100
        ax2.plot(df['Annee'][1:], growth_rates[1:], label='Taux de croissance (%)', 
                linewidth=2, color='#E76F51', alpha=0.7, linestyle='--')
        ax2.set_ylabel('Taux de croissance (%)', color='#E76F51')
        ax2.tick_params(axis='y', labelcolor='#E76F51')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_birth_death_rates(self, df, ax):
        """Plot des taux de natalité et mortalité"""
        ax.plot(df['Annee'], df['Taux_Natalite'], label='Taux de natalité (‰)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        ax.plot(df['Annee'], df['Taux_Mortalite'], label='Taux de mortalité (‰)', 
               linewidth=2, color='#E76F51', alpha=0.8)
        
        ax.set_title('Taux de Natalité et Mortalité (pour 1000 habitants)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Taux (‰)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_age_structure(self, df, ax):
        """Plot de la structure par âge"""
        ax.plot(df['Annee'], df['Part_Moins_20_Ans'] * 100, label='Moins de 20 ans (%)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        ax.plot(df['Annee'], df['Part_Plus_60_Ans'] * 100, label='Plus de 60 ans (%)', 
               linewidth=2, color='#E76F51', alpha=0.8)
        
        # Calculer la part des 20-60 ans
        part_20_60 = [100 - (df.loc[i, 'Part_Moins_20_Ans'] * 100 + df.loc[i, 'Part_Plus_60_Ans'] * 100) 
                      for i in range(len(df))]
        ax.plot(df['Annee'], part_20_60, label='20-60 ans (%)', 
               linewidth=2, color='#F9A602', alpha=0.8)
        
        ax.set_title('Structure de la Population par Âge', fontsize=12, fontweight='bold')
        ax.set_ylabel('Part de la population (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_hdi_evolution(self, df, ax):
        """Plot de l'évolution de l'IDH"""
        ax.plot(df['Annee'], df['IDH'], label='IDH', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        
        ax.set_title('Évolution de l\'Indice de Développement Humain (IDH)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('IDH')
        ax.set_ylim(0.6, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Ajouter des lignes de référence pour les catégories d'IDH
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Développement élevé')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Développement moyen')
        ax.legend()
    
    def _plot_balances(self, df, ax):
        """Plot des soldes naturel et migratoire"""
        ax.bar(df['Annee'], df['Solde_Naturel'], label='Solde naturel', 
              color='#2A9D8F', alpha=0.7)
        ax.bar(df['Annee'], df['Solde_Migratoire'], label='Solde migratoire', 
              color='#E76F51', alpha=0.7, bottom=df['Solde_Naturel'])
        
        ax.set_title('Soldes Naturel et Migratoire', fontsize=12, fontweight='bold')
        ax.set_ylabel('Personnes')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_life_expectancy(self, df, ax):
        """Plot de l'espérance de vie"""
        ax.plot(df['Annee'], df['Esperance_Vie'], label='Espérance de vie', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        
        ax.set_title('Évolution de l\'Espérance de Vie', fontsize=12, fontweight='bold')
        ax.set_ylabel('Années')
        ax.grid(True, alpha=0.3)
    
    def _plot_economic_indicators(self, df, ax):
        """Plot des indicateurs économiques"""
        # PIB par habitant
        ax.plot(df['Annee'], df['PIB_Par_Habitant'], label='PIB par habitant (k€)', 
               linewidth=2, color='#2A9D8F', alpha=0.8)
        
        ax.set_title('Indicateurs Économiques', fontsize=12, fontweight='bold')
        ax.set_ylabel('PIB par habitant (k€)', color='#2A9D8F')
        ax.tick_params(axis='y', labelcolor='#2A9D8F')
        ax.grid(True, alpha=0.3)
        
        # Taux de chômage en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Taux_Chomage'] * 100, label='Taux de chômage (%)', 
                linewidth=2, color='#E76F51', alpha=0.8)
        ax2.set_ylabel('Taux de chômage (%)', color='#E76F51')
        ax2.tick_params(axis='y', labelcolor='#E76F51')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_demographic_projection(self, df, ax):
        """Plot de la projection démographique"""
        years = df['Annee']
        
        # Projection de la population par tranche d'âge
        bottom = np.zeros(len(years))
        categories = ['Part_Moins_20_Ans', 'Part_Plus_60_Ans']
        colors = ['#2A9D8F', '#E76F51']
        labels = ['Moins de 20 ans', 'Plus de 60 ans']
        
        # Calculer la part des 20-60 ans
        part_20_60 = [1 - (df.loc[i, 'Part_Moins_20_Ans'] + df.loc[i, 'Part_Plus_60_Ans']) 
                      for i in range(len(df))]
        
        # Ajouter les trois catégories
        ax.bar(years, df['Part_Moins_20_Ans'] * df['Population'], label='Moins de 20 ans', 
               color='#2A9D8F', alpha=0.7)
        ax.bar(years, part_20_60 * df['Population'], label='20-60 ans', 
               color='#F9A602', alpha=0.7, bottom=df['Part_Moins_20_Ans'] * df['Population'])
        ax.bar(years, df['Part_Plus_60_Ans'] * df['Population'], label='Plus de 60 ans', 
               color='#E76F51', alpha=0.7, 
               bottom=(df['Part_Moins_20_Ans'] * df['Population'] + 
                       np.array(part_20_60) * df['Population']))
        
        ax.set_title('Projection Démographique par Tranche d\'Âge', fontsize=12, fontweight='bold')
        ax.set_ylabel('Population')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _generate_demographic_insights(self, df):
        """Génère des insights analytiques adaptés au territoire"""
        print(f"🏝️ INSIGHTS DÉMOGRAPHIQUES - {self.territoire} (DROM-COM)")
        print("=" * 60)
        
        # 1. Statistiques de base
        print("\n1. 📈 STATISTIQUES GÉNÉRALES:")
        avg_population = df['Population'].mean()
        avg_birth_rate = df['Taux_Natalite'].mean()
        avg_death_rate = df['Taux_Mortalite'].mean()
        avg_hdi = df['IDH'].mean()
        
        print(f"Population moyenne: {avg_population:,.0f} habitants")
        print(f"Taux de natalité moyen: {avg_birth_rate:.1f} ‰")
        print(f"Taux de mortalité moyen: {avg_death_rate:.1f} ‰")
        print(f"IDH moyen: {avg_hdi:.3f}")
        
        # 2. Croissance démographique
        print("\n2. 📊 ÉVOLUTION DÉMOGRAPHIQUE:")
        population_growth = ((df['Population'].iloc[-1] / 
                             df['Population'].iloc[0]) - 1) * 100
        natural_balance = df['Solde_Naturel'].mean()
        migration_balance = df['Solde_Migratoire'].mean()
        
        print(f"Croissance de la population ({self.start_year}-{self.end_year}): {population_growth:.1f}%")
        print(f"Solde naturel moyen: {natural_balance:.0f} personnes/an")
        print(f"Solde migratoire moyen: {migration_balance:.0f} personnes/an")
        
        # 3. Structure par âge
        print("\n3. 👥 STRUCTURE PAR ÂGE:")
        young_share = df['Part_Moins_20_Ans'].mean() * 100
        elderly_share = df['Part_Plus_60_Ans'].mean() * 100
        working_share = 100 - young_share - elderly_share
        
        print(f"Part des moins de 20 ans: {young_share:.1f}%")
        print(f"Part des 20-60 ans: {working_share:.1f}%")
        print(f"Part des plus de 60 ans: {elderly_share:.1f}%")
        
        # 4. Indicateurs de développement
        print("\n4. 📋 INDICATEURS DE DÉVELOPPEMENT:")
        life_expectancy = df['Esperance_Vie'].mean()
        unemployment = df['Taux_Chomage'].mean() * 100
        gdp_per_capita = df['PIB_Par_Habitant'].mean()
        
        print(f"Espérance de vie moyenne: {life_expectancy:.1f} ans")
        print(f"Taux de chômage moyen: {unemployment:.1f}%")
        print(f"PIB par habitant moyen: {gdp_per_capita:.1f} k€")
        
        # 5. Spécificités du territoire
        print(f"\n5. 🌟 SPÉCIFICITÉS DE {self.territoire.upper()}:")
        print(f"Spécialités: {', '.join(self.config['specialites'])}")
        
        # 6. Événements marquants
        print("\n6. 📅 ÉVÉNEMENTS MARQUANTS:")
        print("• 2008-2009: Crise financière mondiale")
        print("• 2011: Départementalisation de Mayotte")
        print("• 2017: Mouvements sociaux en Guyane")
        print("• 2018-2021: Référendums en Nouvelle-Calédonie")
        print("• 2020-2021: Pandémie de COVID-19")
        
        # 7. Recommandations
        print("\n7. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        
        if young_share > 40:  # Population très jeune
            print("• Investir massivement dans l'éducation et la formation")
            print("• Développer des politiques d'emploi pour les jeunes")
            print("• Créer des infrastructures adaptées à une population jeune")
        
        if elderly_share > 25:  # Population vieillissante
            print("• Adapter le système de santé au vieillissement")
            print("• Développer les services aux personnes âgées")
            print("• Favoriser le maintien à domicile")
        
        if unemployment > 15:  # Chômage élevé
            print("• Développer des programmes de formation professionnelle")
            print("• Soutenir la création d'entreprises et l'entrepreneuriat")
            print("• Diversifier l'économie pour créer des emplois")
        
        if self.territoire in ["Mayotte", "Guyane"]:  # Défis spécifiques
            print("• Améliorer l'accès aux services de base")
            print("• Développer les infrastructures de transport")
            print("• Lutter contre l'habitat informel")
        
        if "tourisme" in self.config["specialites"]:
            print("• Développer un tourisme durable et responsable")
            print("• Valoriser le patrimoine culturel et naturel")
            print("• Former les professionnels du tourisme")

def main():
    """Fonction principale pour les DROM-COM"""
    # Liste des DROM-COM
    territoires = [
        "Guadeloupe", "Martinique", "Guyane", "La Réunion", "Mayotte",
        "Saint-Martin", "Saint-Barthélemy", "Saint-Pierre-et-Miquelon",
        "Wallis-et-Futuna", "Polynésie française", "Nouvelle-Calédonie"
    ]
    
    print("🏝️ ANALYSE DÉMOGRAPHIQUE DES DROM-COM (2002-2025)")
    print("=" * 60)
    
    # Demander à l'utilisateur de choisir un territoire
    print("Liste des territoires disponibles:")
    for i, territoire in enumerate(territoires, 1):
        print(f"{i}. {territoire}")
    
    try:
        choix = int(input("\nChoisissez le numéro du territoire à analyser: "))
        if choix < 1 or choix > len(territoires):
            raise ValueError
        territoire_selectionne = territoires[choix-1]
    except (ValueError, IndexError):
        print("Choix invalide. Sélection de La Réunion par défaut.")
        territoire_selectionne = "La Réunion"
    
    # Initialiser l'analyseur
    analyzer = DromcomDemographyAnalyzer(territoire_selectionne)
    
    # Générer les données
    demographic_data = analyzer.generate_demographic_data()
    
    # Sauvegarder les données
    output_file = f'{territoire_selectionne}_demographic_data_2002_2025.csv'
    demographic_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(demographic_data[['Annee', 'Population', 'Taux_Natalite', 'Taux_Mortalite', 'IDH']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse démographique...")
    analyzer.create_demographic_analysis(demographic_data)
    
    print(f"\n✅ Analyse démographique de {territoire_selectionne} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Démographie, natalité, mortalité, IDH, économie")

if __name__ == "__main__":
    main()