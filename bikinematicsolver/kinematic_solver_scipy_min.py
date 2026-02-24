#Library imports
import numpy as np
import scipy as sp
from scipy.optimize import minimize 
import networkx as nx

#pylint: disable = import-error
from bikinematicsolver.dtypes import Pos

class Kinematic_Solver_Scipy_Min():
    def __init__(self,n_steps):

        self.n_steps = n_steps

    def solve_suspension_motion(self, travel, points, links, kin_loop_points, end_eff_points, secondary_loop_points=None): 

        klp_off, klp_ss, eep_ss, eep_posn = self.get_solution_space_vectors(points, kin_loop_points, end_eff_points, links)
        
        # Convert secondary loop if 6-bar is detected
        sec_ss = None
        if secondary_loop_points:
            sec_klp = np.array([points[name].pos for name in secondary_loop_points], dtype=float)
            sec_ss = self.cartesian_to_link_space(sec_klp, 'loop')

        input_angles = self.find_input_angle_range(travel, klp_off, klp_ss, eep_ss, eep_posn, points, 
                                                   end_eff_points, kin_loop_points, sec_ss)
        
        point_results = np.zeros((len(kin_loop_points) + len(end_eff_points), 2, input_angles.shape[0]))
        
        for i in range(len(input_angles)):
            klp_ss[0] = input_angles[i]
            klp_sol = self.solve_kinematic_loop(klp_ss, sec_ss)
            point_results[:, :, i] = self.solution_to_cartesian(klp_off, klp_sol, eep_ss, eep_posn)

        # Map results back to point names
        points_list = kin_loop_points + end_eff_points
        solution = {name: Pos(point_results[i, 0, :], point_results[i, 1, :]) for i, name in enumerate(points_list)}
        return solution
    
    def get_solution_space_vectors(self,
        points,
        kin_loop_points,
        end_eff_points,
        links):
        """
        Returns vectors in solution space form, ready for solving. This form is the following:

        1st return: klp_off - a point of form [x,y], denoting the offset of the linkage loop origin from the
        origin used in the cartesian representation of the bike (currently the bottom left screen pixel)

        2nd return: klp_ss - a vector containing the angle and length generalised coords of the linkage of the format:
        [th12,th23,...,th(n-1)(n),th(n)1 , L12,L23,...,L(n-1)(n),L(n)1], i.e the magnitude and angle of the vectors
        between the points of the loop.

        Note - These are ordered with respect to self.kinematic_loop_points list, where in the notation used
        in this fcn description the first entry (self.kinematic_loop_points[0]) will be the name of the point 
        with coords [x1,y1]

        3rd return: eep_ss - a vector containing the angle and length offset of a particular end effector from
        linkage point given in eep_posn. This vector has form [th,L]

        4th return: eep_posn - the index of the linkage point that the end effector is offset from

        Note - These are ordered with respect to self.end_eff_points list, where in the notation used
        in this fcn description the first entry (self.end_eff_points[0]) will be the name of the end effector 
        with offset given by eep_ss[0] from linkage point self.kinematic_loop_points(eep_posn[0]) 

        """
        klp = np.array([points[name].pos for name in kin_loop_points],dtype=float) #vector of points in form [[x1,y1],[x2,y2],...,[xn,yn]]
        eep = np.array([points[name].pos for name in end_eff_points],dtype=float) #vector of points in form [[x1,y1],[x2,y2],...,[xn,yn]]

        #Convert loop 
        klp_off = klp[0,:] 
        klp_ss = self.cartesian_to_link_space(klp,'loop')
        
        #Convert end effectors
        eep_posn=[]
        eep_ss =np.zeros(eep.shape[0]*2) # Converting (n x 2) [[x1,y1]...[xn,yn]] shape to [th1...thn,L1...Ln] (2n x 1) shape

        for end_eff_index in range(len(end_eff_points)): #Loop through end eff points and find attachment point and offset            
            
        # 1. Get the LIST of attachment points
            attach_point_list = self.find_end_eff_attach_point(
                end_eff_points[end_eff_index],
                links,
                kin_loop_points)
                
            # 2. THE FIX: Intelligently pick the correct base point of the physical link!
            if len(attach_point_list) == 2:
                u = attach_point_list[0]
                v = attach_point_list[1]
                n_loop = len(kin_loop_points)
                
                # Check which point connects to the other in the correct loop sequence
                if (u + 1) % n_loop == v:
                    attach_point_index = u
                elif (v + 1) % n_loop == u:
                    attach_point_index = v
                else:
                    attach_point_index = attach_point_list[0] # Fallback
            else:
                attach_point_index = attach_point_list[0] # Faux-bars / single pivot fallback       
            
            # 3. Find offset from attach point to end effector
            # Because attach_point_index is now an integer, this array stays perfectly flat
            Th,L = self.cartesian_to_link_space([klp[attach_point_index],eep[end_eff_index]])
            Th = klp_ss[attach_point_index] - Th #Find constant offset from link, orignal Th was theta from global x and non-constant!!!!!

            #Store in expected format
            eep_posn.append(attach_point_index)
            eep_ss[end_eff_index] = Th
            eep_ss[eep.shape[0]+end_eff_index] = L # Converting (n x 2) [[x1,y1]...[xn,yn]] shape to [th1...thn,L1...Ln] (2n x 1)  shape

        return klp_off,klp_ss,eep_ss,eep_posn

    # def find_end_eff_attach_point(self,
    #     end_eff_point,
    #     links,
    #     kin_loop_points):
    #     """
    #     Returns index of linkage point attachment (via link) for given end_eff point. Finds first link in kinematic loop (lowest index) as 
    #     this follows for convention of link angle indexing later. Needs error checking written if no attachment at all.
    #     """
    #     possible_links = []
    #     for link in links.values():   
    #         if link.a == end_eff_point:
    #             possible_links.append(kin_loop_points.index(link.b))
    #         if link.b == end_eff_point:
    #             possible_links.append(kin_loop_points.index(link.a))
        
    #     return min(possible_links)
    # def find_end_eff_attach_point(self, eff_name, kin_loop_points, links):
    #     possible_links = []
    #     for name, link in links.items():
    #         if link.a == eff_name:
    #             # SAFETY CHECK: Ensure the anchor is actually in this loop
    #             if link.b in kin_loop_points: 
    #                 possible_links.append(kin_loop_points.index(link.b))
    #         elif link.b == eff_name:
    #             # SAFETY CHECK
    #             if link.a in kin_loop_points:
    #                 possible_links.append(kin_loop_points.index(link.a))
    #     return possible_links
    def find_end_eff_attach_point(self, eff_name, arg1, arg2):
        # Auto-detect which argument is the dictionary of links
        if isinstance(arg1, dict) or (isinstance(arg1, list) and len(arg1) > 0 and hasattr(arg1[0], 'a')):
            links_data = arg1
            kin_loop_points = arg2
        else:
            links_data = arg2
            kin_loop_points = arg1

        possible_links = []
        # Safely extract the link objects
        link_objects = links_data.values() if isinstance(links_data, dict) else links_data
        
        for link in link_objects:
            if link.a == eff_name:
                if link.b in kin_loop_points: 
                    possible_links.append(kin_loop_points.index(link.b))
            elif link.b == eff_name:
                if link.a in kin_loop_points:
                    possible_links.append(kin_loop_points.index(link.a))
                    
        return possible_links

    # def solve_kinematic_loop(self,loop_ls):
    #     """
    #     Expects (2n x 1) input vector of form v = [th1,...,th(n),L1,...,L(n)]. Typical usage is to set the input angle,
    #     th1 to desired value, then pass to this function to find new solution vector for this input angle.

    #     Returns (2n x 1) solution vector s = [th1,...,th(n),L1,...,L(n)] satisfying the linkage constraint equation
    #     """
    #     #Process input data for solver

    #     mid = int(loop_ls.shape[0]/2)

    #     if mid <= 2:
    #         return loop_ls
        
    #     x = loop_ls[1:mid-1] #Constrained coordinates to be found by optimiser (this defo works for 4-bar need to test higher dims...)

    #     geo = np.vstack([loop_ls[0],loop_ls[mid-1:]]) #Constant generalised coords (Link lengths, ground angle)
    #     x = x.flatten()

    #     #Solve by minimising error in linkage constraint equation
    #     #THIS IS THE LINE THAT BREAKS - HERE
    #     res = sp.optimize.minimize(self.constraint_eqn,
    #                                x,
    #                                geo) #This solves by minimsing error in the linkage loop equation
    #     #Return solution in expected format
    #     x_sol = np.vstack(res.x)
    #     sol = loop_ls
    #     sol[1:mid-1] = x_sol
    #     return sol
    # In kinematic_solver_scipy_min.py

    def solve_kinematic_loop(self, loop_ls, secondary_ls=None):
        """
        Handles Single Pivot, 4-Bar, and 6-Bar simultaneous minimization.
        """
        mid = int(loop_ls.shape[0] / 2)
        if mid <= 2: # Single Pivot bypass
            return loop_ls

        # Partition variables for Loop 1
        x0 = loop_ls[1:mid-1].flatten()
        
        # THE FIX: Properly separate start angle, end (frame) angle, and lengths
        geo = [ [loop_ls[0], loop_ls[mid-1], loop_ls[mid:]] ] 
        
        # Add variables for Loop 2 if 6-bar/Shock Loop is present
        if secondary_ls is not None:
            mid2 = int(secondary_ls.shape[0] / 2)
            x_sec = secondary_ls[1:mid2-1].flatten()
            x0 = np.concatenate([x0, x_sec])
            geo.append([secondary_ls[0], secondary_ls[mid2-1], secondary_ls[mid2:]])

        # Simultaneous minimization of all loop constraints
        res = sp.optimize.minimize(self.constraint_eqn, x0, args=(geo,))
        
        # Reconstruct the primary loop solution
        sol = loop_ls.copy()
        sol[1:mid-1] = res.x[:mid-2].reshape(-1, 1)
        return sol
    
    # def constraint_eqn(self,x,args):
    #     """
    #     Finds vector u = [u_x,u_y], given by u_x = sum(lcos(th)), and u_y = sum(lsin(th)) by some neat matrix multiplication
    #     Then finds magnitude of this vector and returns it -> this signifies the error in the linkage constraint
    #     """
    #     #Data setup
    #     geo = args
    #     n = len(x)+len(args)
    #     q = int(n/2)
    #     theta = np.vstack([geo[0],np.reshape(x,(len(x),1)),geo[1:q-len(x)]])
    #     theta = theta.transpose()

    #     #Constraint eqn
    #     ctheta = np.cos(theta)
    #     stheta = np.sin(theta)
    #     thetas = np.vstack([ctheta,stheta])
    #     L = args[q-len(x):]
    #     u = thetas @ L #matrix mult
    #     #Error
    #     err = np.linalg.norm(u)

    #     return err
    def constraint_eqn(self, x, geo_args):
        """
        Calculates error for multiple loops. Error is zero only when ALL loops close.
        """
        # Loop 1 extraction
        geo1 = geo_args[0]
        lengths1 = geo1[2] # THE FIX: Look at index 2 for the lengths array
        n1_unknowns = len(lengths1) - 2 # derived from link count
        x1 = x[:n1_unknowns]
        err1 = self._calc_loop_error(x1, geo1)

        # Loop 2 extraction (if 6-bar)
        if len(geo_args) > 1:
            geo2 = geo_args[1]
            lengths2 = geo2[2] # THE FIX: Look at index 2
            n2_unknowns = len(lengths2) - 2
            x2 = x[n1_unknowns : n1_unknowns + n2_unknowns]
            err2 = self._calc_loop_error(x2, geo2)
            # Combine errors: both must be zero
            return np.linalg.norm([err1, err2])
        
        return err1

    def _calc_loop_error(self, x_angles, geo):
        """
        Helper function to calculate the tip-to-tail vector sum error.
        Calculates how far the loop is from closing physically.
        """
        start_angle = geo[0]
        end_angle = geo[1]
        lengths = geo[2]
        
        # THE FIX: Rebuild full theta vector using the true frame angle, not [0]
        theta = np.vstack([start_angle, np.reshape(x_angles, (-1, 1)), end_angle]) 
        
        # Standard vector loop closure: calculate X and Y components
        u_x = np.sum(lengths.flatten() * np.cos(theta.flatten()))
        u_y = np.sum(lengths.flatten() * np.sin(theta.flatten()))
        
        # Return the hypotenuse (the absolute distance error) as a single scalar
        return np.sqrt(u_x**2 + u_y**2)
    # def find_input_angle_range(self,
    #     travel,
    #     klp_off,
    #     klp_ss,eep_ss,
    #     eep_posn,points,
    #     end_eff_points,
    #     kin_loop_points):
    #     """
    #     Takes desired simulation travel and solution space vectors, and returns a range of input angles from [th0,...,tht], where th0 is the starting angle
    #     at zero suspension travel, and tht is the angle of the input link that gives the desired simulation travel. The number of angles in the 
    #     range is currently hardcoded at 100, but I will change this at some point.

    #     Currently no error checking for unachievable angles - needs implemented likely based off whether optimisation target < 1e-2 or something similar  
    #     """
    #     #Find rear wheel initial vertical position
    #     for name,point in points.items():
    #         if point.type == 'rear_wheel':
    #             rear_wheel_name = name
    #             rear_wheel_init_y = point.pos[1]
    #     if rear_wheel_name in end_eff_points:
    #         r_w_ind = end_eff_points.index(rear_wheel_name) + len(kin_loop_points) #List index of rear wheel point coordinates
    #     if rear_wheel_name in kin_loop_points:
    #         r_w_ind = kin_loop_points.index(rear_wheel_name)
    #     #Setup up solver to find angle that minimises error between desired y position (at specified travel), and y position of rear wheel
    #     #found from linkage solver
    #     desired_y = rear_wheel_init_y+travel
    #     th_in_0 = float(klp_ss[0])

    #     res = sp.optimize.minimize(
    #         self.travel_find_eqn,
    #         th_in_0,
    #         [desired_y, r_w_ind, klp_off, klp_ss, eep_ss, eep_posn],
    #         method = 'Nelder-Mead',
    #         options = {'disp':False})
    #     #Create return vector from initial and final angles
    #     th_in_end = res.x
    #     input_angles = np.linspace(th_in_0,th_in_end,num=self.n_steps)
    #     return input_angles

    def find_input_angle_range(self, max_travel, klp_off, klp_ss, eep_ss, eep_posn, points, end_eff_points, kin_loop_points, sec_ss=None):
        """
        Finds the start and end angles of the driving link to achieve the desired wheel travel.
        """
        # 1. Find the index of the Rear Wheel in your output array
        points_list = kin_loop_points + end_eff_points
        
        # Safely find the rear wheel name by inspecting the points dictionary
        rw_name = None
        for name, pt in points.items():
            if pt.type == 'rear_wheel':
                rw_name = name
                break
                
        # Fallback just in case the JSON type is missing
        if rw_name is None:
            rw_name = 'Rear_Wheel'
            
        rw_idx = points_list.index(rw_name)

        # 2. Find the static (0 travel) starting Y position
        klp_sol_start = self.solve_kinematic_loop(klp_ss.copy(), secondary_ls=sec_ss)
        cart_start = self.solution_to_cartesian(klp_off, klp_sol_start, eep_ss, eep_posn)
        start_y = cart_start[rw_idx, 1]
        
        theta_start = klp_ss[0].copy() # The initial angle from your geometry file

        # 3. Find the angle for Maximum Travel
        # We use a minimizer to find the input angle that gives us the max vertical displacement
        res = sp.optimize.minimize(
            self.travel_find_eqn, 
            x0=theta_start + 0.1, # Initial guess (rotate the link slightly)
            args=(max_travel, klp_off, klp_ss.copy(), eep_ss, eep_posn, start_y, rw_idx, sec_ss),
            method='Nelder-Mead' # Robust for 1D searches
        )
        
        theta_end = res.x[0]

        # 4. Generate the array of angles for the solver to step through
        input_angles = np.linspace(theta_start, theta_end, self.n_steps)
        
        return input_angles
        
    # def travel_find_eqn(self,x,args):
    #     """
    #     Solves the linkage equation with the input solution space vectors, and returns the (absolute!) error between the rear wheel y position and the desired.

    #     Maybe need to look at the fcn input to make more clear, but last time it tried it didn't work so well with the sp.optimize.minimize this is passed to
    #     """
    #     #This is a bit ugly for now, maybe find a neater way to pass through the variables??

    #     desired_y = args[0]

    #     r_w_ind = args[1]

    #     klp_off = args[2]

    #     klp_ss = args[3]

    #     eep_ss = args[4]

    #     eep_posn = args[5]


    #     klp_ss[0]= x #The optimisation variable is the input angle of the linkage
    #     # If it's a single pivot, we don't need to 'solve' the loop, 
    #     # the positions are determined solely by klp_ss[0]
    #     if int(klp_ss.shape[0]/2) <= 2:
    #         klp_sol = klp_ss
    #     else:
    #         klp_sol = self.solve_kinematic_loop(klp_ss) #Solve linkage with this angle

    #     #Convert to cartesian and find error between desired and actual rear wheel y position
    #     sol_cartesian = self.solution_to_cartesian(
    #         klp_off,
    #         klp_sol,
    #         eep_ss,
    #         eep_posn)

    #     y = sol_cartesian[r_w_ind,1]
    #     err = np.abs(desired_y-y)
    #     #print(err)
    #     return err

    def travel_find_eqn(self, theta_in, target_travel, klp_off, klp_ss, eep_ss, eep_posn, start_y, rw_idx, sec_ss=None):
        """
        Evaluates the difference between the current wheel Y position and the target Y position.
        """
        # 1. Set the trial input angle
        klp_ss[0] = theta_in
        
        # 2. Solve the linkage (Handles both 4-bar and 6-bar seamlessly)
        klp_sol = self.solve_kinematic_loop(klp_ss, secondary_ls=sec_ss)
        
        # 3. Convert back to cartesian coordinates
        cart_pts = self.solution_to_cartesian(klp_off, klp_sol, eep_ss, eep_posn)
        
        # 4. Get the current Y position of the rear wheel
        current_y = cart_pts[rw_idx, 1]
        
        # 5. Calculate vertical travel achieved
        current_travel = current_y - start_y
        
        # 6. Return the error (SciPy will try to make this 0)
        return np.abs(current_travel - target_travel)

    def solution_to_cartesian(self,
        klp_off,
        klp_sol,
        eep_ss,
        eep_posn):
        """
        Takes klp_off,klp_sol,eep_ss,eep_posn as described in self.get_solution_space_vectors, and returns cartesian coords of form:
        [[xl1,yl1],...,[xl(nl),yl(nl)],[xe1,ye1],...,[xe(ne),ye(ne)]], where nl and ne denote number of kinematic loop and end effector
        points respectively
        """
        #Linkage loop points can be directly converted
        klp_v = self.link_space_to_cartesian(
            klp_off,
            klp_sol,
            'loop')
        
        #End effector points need dealt with 
        mid = int(eep_ss.shape[0]/2)
        eep_v = np.zeros((mid,2)) #Reshape (2n x 1)-> (n x 2)
        for i in range(mid): #Loop through ee generalised coords and get position in cartesian space
            pos = self.link_space_to_cartesian(
                klp_v[eep_posn[i],:], #Offset is attach point
                np.vstack([klp_sol[eep_posn[i]]-eep_ss[i],eep_ss[mid+i]])) #Gets representation in form [th(n),L(n)]. Global th(n) must be found from global 
                                                                                                          #klp theta, klp_sol[eep_posn[i]], minus the constant offset, eep_ss[i],
                                                                                                          # the end effector has form the kinematic link
            eep_v[i,:]=pos[1] #Don't need attachemnt coords, only end effector coords

        #Return expected format
        return np.vstack([klp_v,eep_v])

    ##Coordinate conversion functions:
    def cartesian_to_link_space(self,v,*params):
        """
        Takes a (n x 2) vector of points in form v = [[x1,y1],[x2,y2],...,[x(n),y(n)]], and converts to generalised coord: angles from horizontal Theta,
        and magnitdues L, measured from  successive points/joints. Return is (2(n-1) x 1) vector of form [th12,th23,...,th(n-1)(n),L12,L23,...,L(n-1)(n)]
        If 'loop' is passed as a parameter, the generalised coord to return from the last point in v to the first is also included, returning a (2n x 1) vector
        of form [th12,th23,...,th(n-1)(n),th(n)1,L12,L23,...,L(n-1)(n),L(n)1]
        """

        #Add first point to end of list again if 'loop'is specified
        if 'loop' in params:
            v = np.concatenate([v,[v[0,:]]])

        #Perform conversion
        diff = np.diff(v,axis=0)
        Theta = np.vstack(np.arctan2(diff[:,1],diff[:,0])) #Vector of angles [th1,th2,...,thn]
        L = np.vstack(np.linalg.norm(diff,ord = 2,axis=1)) #Vector of lengths [L1,L2,...,Ln]
        ls = np.vstack([Theta,L])
        return ls

    def link_space_to_cartesian(self,offset,ls,*params):
        """
        Takes set of link space generalised coordinates of form [th12,th23,...,th(n-1)(n),L12,L23,...,L(n-1)(n)], and an offset coordinate [x0,y0] and 
        returns cartesian coords of form [[x1,y1],[x2,y2],...,[x(n),y(n)]]. For loops use the 'loop' in *params otherwise it will return the origin of the loop
        twice - at the start and the end.
        """
        #Data sizing fun
        N = int(ls.shape[0]/2)
        Theta = ls[0:N]
        L = ls[N:]
        n = Theta.shape[0]
        #Shape return vector depending on loop or not
        if 'loop' in params:
            v = np.zeros((n,2)) #Vector of points in form [[x1,y1],[x2,y2],...,[xn,yn]]
        else:
            v = np.zeros((n+1,2)) #Vector of points in form [[x1,y1],[x2,y2],...,[xn,yn]]

        #Conversion
        v[0,:] = np.array([0,0]) + offset #First point is offset (some weird numpy stuff going on here as well :) )
        for i in range(0,v.shape[0]-1): #Loop through -> next coords are [xold,yold] + [lcos(th),Lsin(th)]
            Lcos = L[i]*np.cos(Theta[i])
            Lsin = L[i]*np.sin(Theta[i])
            v[i+1,:] = v[i,:] + np.hstack([Lcos,Lsin]) #loop to find cartesian coords
        
        return v
