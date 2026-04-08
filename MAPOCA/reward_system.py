# Ce fichier nous permet de calculer les récompenses pour les agents dans notre environnement de simulation.

import cv2
import numpy as np

class puck_position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get_puck_position(self, observation):
        #conversion en format OpenCV
        image  = observation.astype(np.uint8)

        # on définit ensuite la couleur du palet

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([50, 50, 50])

        # on créer ensuite un masque où on met le palet en blanc et tout ce qui ne l'est pas en noir

        mask = cv2.inRange(image, lower_black, upper_black)

        # on cherche ensuite les contours (les formes blanches)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
                # On prend le plus gros contour (pour éviter le bruit)
                largest_contour = max(contours, key=cv2.contourArea)
                # Calcul du centre de masse du palet
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return cX, cY
        else:
            return None
        

class RewardSystem:
    def __init__(self, goal_x_position):
        self.goal_x = goal_x_position # Position du but adverse
        self.detector = puck_position(0, 0)

    def calculate_integrated_reward(self, observation, agent_pos_ram, game_reward):
        # 1. On cherche le palet visuellement
        puck_coords = self.detector.get_puck_position(observation)
        
        if puck_coords:
            pX, pY = puck_coords
            aX, aY = agent_pos_ram # On peut mixer les deux méthodes !
            
            # Distance Reward
            dist = np.sqrt((aX - pX)**2 + (aY - pY)**2)
            dist_rew = 1.0 / (dist + 1.0)
            
            # Integrated Reward
            return (game_reward * 50) + (dist_rew * 2.0)
        
        return game_reward # Si on perd le palet de vue

    