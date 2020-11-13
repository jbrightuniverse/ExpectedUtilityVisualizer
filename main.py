"""
  Python program with PyGame to represent an N-ary lottery of (N-1) - dimensional space in a 2-dimensional grid without loss of data.

  By James Yu, 2020
"""

import colorsys
import math
import sys
import time

import pygame as pg
from pygame.locals import *

import sympy
from sympy import *

"""
  A semi-brief explanation of the representation:

  A lottery is a set of probabilities on an outcome space such that each outcome has a probability of occurrence, and all probabilities sum to 1.

  In one dimension, we have two outcomes, so this is just a line.
  In two dimensions, we have three outcomes, so we have a triangle of possible outcomes as each point on the previous line now has an additional point to connect to, each with a range from 0 to 100% likelihood

  In three dimensions, this becomes more complex. Following the additive algorithm, we add a point and link all existing points to it. This forms a triangular pyramid, which is difficult to represent.

  In four and higher dimensions, this becomes a type of hyperpyramid. This is normally impossible to render. But we do exactly that here.

  A lottery of N outcomes can be factored out into a compound lottery of a lottery of N-1 outcomes and a lottery of 1 outcome. This is due to the fact that we are linear in probabilities and therefore are able to use the independence axiom to break a lottery into a linear combination of two sub-lotteries.

  For example, take (1/3, 1/3, 1/3). This can be split into a 1/3 probability of 1 and a 2/3 probability of two other equally likely outcomes, within which each would have probability 1/2.
  Note that 1/2 * 2/3 = 1/3, so when the lottery is reduced, we get the original lottery. By varying the 2/3-1/3 split, we can vary the impact by which the third outcome biases the other two.

  By recursively breaking this down, we are able to represent any (N-1)-dimensional lottery as a series of (N-1) chained lines, where the subsequent (N-2) lines and a point on the (N-2)th line are all adjustable with repsect to being linked to the previous line.

  If this is confusing, run the program. You will see a 2-dimensional 3-ary lottery, shaped like a triangle, where all points on the triangle are accessible just by moving the two sliders. This is exactly the recursive representation mentioned: we have a set of lotteries between outcome one and 2 alone, and each lottery has a set of lotteries between it and 3.

  So we can now represent any set of lotteries on a set of outcomes in two dimensions by recursively applying chained dynamic lines.

  Keep reading the program for explanations of indifference fields and risk aversion.
"""

"""
  First, we prepare global variables. These are used to determine various parameters.
"""

# launch the PyGame window
pg.init()
size = width, height = 800, 600
screen = pg.display.set_mode(size)

# variables for:
outcomes = [1, 2, 3] # the outcomes of the lottery
aversion = 143       # person's risk aversion, scaled to coordinates
rdensity = 340       # density of (N-1)>2 dimensional renderings, scaled to coordinates 

# prep some convenient fonts
font = pg.font.Font("OpenSansEmoji.ttf", 20)
font3 = pg.font.Font("OpenSansEmoji.ttf", 15)

# seg: fraction by which to divide the screen to center the graph
seg = len(outcomes)//2 + 1
# offset: base variable to align the graph to the screen
offset = (width/(seg+1))//2

sliders = [] # coordinates of the sliders on the graph
points = []  # coordinates of the vertices/outcomes on the graph

# default probabilities for the given outcomes such that each is equally likely; these probabilities are fractions of the graph lines, not the final reduced lottery probabilities
probabilities = [1/n for n in range(2, len(outcomes)+1)]

inputs = ["", "", "x"] # text inputs, for add/remove/utility function
utilfunc = "x"         # the active utility function
hovering = 0           # the button we are hovering over, if any
cur = -1               # the currently active text box, if any
index = -1             # the currently active slider, if any

base = None # variable for storing the indifference field rendering
changed = False # variable for storing whether the indifference field has been updated or not

# function for scaling a value from one range to another
def nmap(num, low, high, nlow, nhigh):
  if high == low: return nlow # catch a div by zero case
  return (((num - low) * (nhigh - nlow)) / (high - low)) + nlow

# function for drawing a box of text
def textbox(x, y, text, highlight, long = False):
  # select length based on long parameter
  length = [60, 120][long] 
  box = pg.Rect(x, y, length, 20) # draw a box that long
  pg.draw.rect(screen, (128, 128, 128), box) # make it grey
  pg.draw.rect(screen, [(66, 206, 245), (136, 255, 136)][highlight], box, 2) # draw an outline with color based on highlight
  if text:
    # draw text if it exists and offset it to not overlap with the box
    screen.blit(font3.render(text, True, (255, 255, 255)), (x+2,y))

# function for determining if an (x, y) coord intersects a region
def intercept(x, y, x2, y2, width = 60, height = 20):
  return x in range(x2, x2 + width) and y in range(y2, y2 + height)

# function that renders the graph
def render():
  global sliders, points, base, changed

  """
    The graph of the lottery is constructed by the following algorithm:
      - given a set of outcomes, space them alternating in two parallel rows, forming either a trapezoid or parallelogram
      - draw a line between the first two outcomes, and place a dynamic point on the line
      - for every additional outcome, draw a line between the previous dynamic point and the outcome; place a new dynamic point on the line
      - the location of the dynamic point is determined by the fraction stored in the probabilities variable
  """

  # if we have an indifference field, use the last slider as a reference for the indifference surface
  if base: oldslide = sliders[-1]
  else: oldslide = None

  # clear the coordinate variables, clear the screen
  sliders = []
  points = []
  screen.fill((255, 255, 255))

  # convert the stringed utility function to an actual function
  utilityfunc = sympify(utilfunc)
  # determine the actual utility given outcomes by substituting x  in the function for each outcome value
  adjoutcomes = [float(utilityfunc.subs(symbols("x"), a)) for a in outcomes]

  # compute the expectation, variance, and reduced lottery
  ps = probabilities[:1] # take the first probability

  # basic expected utility variable: probability is a fraction of line length; the higher the probability, the higher the chance of the rightmost outcome on the line, so it is the probability of the second of two outcomes, and (1-p) is the probability of the first
  ev = (1-ps[0]) * adjoutcomes[0] + ps[0] * adjoutcomes[1]
  ps.insert(0, 1-ps[0]) # put 1-p into the probability list too

  # for all other outcomes, recursively 
  for j in range(1, len(outcomes)-1):
    xprob = probabilities[j] # take the next probability
    ps = [(1-xprob) * a for a in ps] # adjust all previous probabilities by (1-p) since this is on the left side of whatever line
    ps.append(xprob) # add the new probability
    ev = (1-xprob) * ev + xprob * adjoutcomes[j+1] # do the same to the expected utility
  
  # compute the variance
  variance = sum([p*(a-ev)**2 for p, a in zip(ps, adjoutcomes)])

  # if the indifference fields were pre-computed, render it and the indifference field relative to the current expected utility
  if base:
    surf = base.copy()
    temp = pg.PixelArray(surf)
    # adjust for risk aversion; explanation later on when this is done again
    aver = nmap(aversion, 115, 171, -1, 1)

    # if its negative, we are risk loving
    if aver < 0:
      aver = abs(aver)
      sign = -1
    else: sign = 1

    # compute corresponding color in HSL form
    h = nmap((1-aver)*ev - sign*aver*variance, min(adjoutcomes), max(adjoutcomes), 0, 276)
    # convert to RGB
    col = colorsys.hls_to_rgb(h/360, 0.5, 1)
    col = (int(round(255*col[0])), int(round(255*col[1])), int(round(255*col[2])))
    # replace all points on the indifference field image of that color with grey to represent indifference field
    temp.replace(col, (128, 128, 128), 0.01)
    del temp
    screen.blit(surf, (0,0))

  # the text at the top of the screen
  screen.blit(font3.render("Click a slider and move your mouse horizontally to change the probabilities. Move slower for higher precision.", True, (0,0,0)), (0,0))
  screen.blit(font3.render("Click again to save. Use backward induction to control each individual probability. Utility grows red->blue->red.", True, (0,0,0)), (0,15))
  screen.blit(font3.render(f"The gold slider represents the final probability coordinate in {len(outcomes)-1}-dimensional space.", True, (0,0,0)), (0,30))
  screen.blit(font3.render("Risk Aversion:LO               HI   Render Density: HI                 LO   The grey shades represent the indifference set.", True, (0,0,0)), (0,46))

  # shapes needed at the top of the screen 
  pg.draw.line(screen, (0,0,0) ,(115, 55), (170, 55), 1)
  pg.draw.line(screen, (0,0,0) ,(340, 55), (400, 55), 1)
  pg.draw.circle(screen, (0,255,0), (aversion, 55), 3)
  pg.draw.circle(screen, (0,255,0), (rdensity, 55), 3)
  pg.draw.rect(screen, (240, 84, 70), pg.Rect(0, 80, 370, 20))

  # text entry text
  screen.blit(font3.render("Add:", True, [(255, 255, 255), (136, 255, 136), (255, 255, 255), (255, 255, 255)][hovering]), (0, 80))
  screen.blit(font3.render("Remove:", True, [(255, 255, 255), (255, 255, 255), (136, 255, 136), (255, 255, 255)][hovering]), (92, 80))
  screen.blit(font3.render("Utility:", True, [(255, 255, 255), (255, 255, 255), (255, 255, 255), (136, 255, 136)][hovering]), (217, 80))
  screen.blit(font3.render("Click a box to enter text, click a word to submit.", True, (0,0,0)), (width//2,80))

  # metrics
  densityx = nmap(rdensity, 340, 400, 2, 8) 
  try: densityx = max(3, int(round(200/((len(outcomes)-2)**densityx)))) # density computation, used later
  except: densityx = "none"
  screen.blit(font3.render(f"Min. 2 outcomes; risk aver. = {round(nmap(aversion, 115, 171, -1, 1), 3)}, density = {densityx}", True, (0,0,0)), (0, 100))

  # text boxes
  textbox(32, 80, inputs[0], cur==0)
  textbox(157, 80, inputs[1], cur==1)
  textbox(265, 80, inputs[2], cur==2, True)

  # some variables
  save = None # variable for if we have a slide
  gold = None # variable for the final slider
  prev = None # previously seen slider (or the first vertex)
  frst = None # the first vertex, which needs to be redrawn
  todraw = [] # all the sliders, which need to be redrawn

  # iterate over all outcomes
  for i in range(len(outcomes)):
    # compute width and height of each outcome vertex and its label
    wd = width//(seg+1) * ((i//2) +1)
    hd = 3*height//4
    hy = hd + 20
    if i % 2 == 1: # if it is odd-indexed it goes on top with offset
      wd += int(round(offset))
      hd = height//4
      hy = hd - font.size(str(outcomes[i]))[1] - 12

    # if this is not the first loop, we draw a slider
    if prev:
      # given probabilities, compute the coordinates of the slider based on common ratio of line length to x and y
      cx = wd - (1-probabilities[i-1])*(wd-prev[0])
      cy = hd - (1-probabilities[i-1])*(hd-prev[1])
      mid = [int(round(cx)), int(round(cy))]
      sliders.append(mid)
      pg.draw.line(screen, (0,0,0), prev, (wd, hd), 6) # draw the line
      if index == i-1: 
        # if index (the clicked point variable) is this one, save the slider so we can color it green for visibility
        save = mid
      if i == len(outcomes)-1: 
        # if this is the last one, its the gold slider
        gold = mid
      todraw.append(mid)

    if i == 0: 
      # if this is the first one, we set previous to the first outcome
      prev = [wd, hd]
      frst = prev
    else: prev = mid # otherwise its the current slider
    points.append((wd, hd))
    screen.blit(font.render(str(outcomes[i]), True, (0,0,0)), (wd, hy)) # draw the label
    pg.draw.circle(screen, (91, 104, 199), (wd, hd), 16) # draw the vertex

  # redraw the first vertex
  pg.draw.circle(screen, (91, 104, 199), frst, 16) 
  # redraw the sliders
  for entry in todraw:
    pg.draw.circle(screen, (235, 70, 98), entry, 8)

  # recolor the final slider
  pg.draw.circle(screen, (212,175,55), gold, 8)

  if save:
    # if we clicked on a slider, color it green so we can see
    pg.draw.circle(screen, (136, 255, 136), save, 8)

  ignore = False
  # if we haven't yet computed the indifference fields, do that now
  if not changed:
    changed = True
    ignore = True # this tells us not to render the result of the function when we finish

    """
      In an (N-1)-dimensional lottery, indifference curves become indifference fields of N-2 dimensions. Since we have an expected utility form, these are parallel constructs.

      We compute these by calculating the expected utility at as many points as possible and then scaling them from their range of values to a new range based on color. We then draw these colors onto an image for saving and using later.

      Points on the image of the same color represent an indifference field i.e. points with the same expected utility.

    """

    # prep the image
    base = pg.Surface((800, 600))
    base.fill((255, 255, 255))

    # recursive function to go through each dimension of the lottery, branch out to all possible probability combinations, and color them in
    # normally this would result in a severe amount of overlapping points; we resolve this using variable render density i.e. the number of entries/probabilities between 0 and 1 we use to render
    def shaderecursive(level, probability, prev, probabilities):
      # compute the coordinates of the next outcome/vertex
      wd = points[level+1][0]
      hd = points[level+1][1]

      # scale the provided density variable
      sdensity = nmap(rdensity, 340, 400, 2, 8) 
      try: density = max(3, int(round(200/((len(sliders)-1)**sdensity)))) # try to convert it to a number of points; if it fails, we have division by zero because there are no points to color
      except: return

      # for all possible probabilities...
      for k in range(0, density):
        # change density to a value between 0 and 1
        prob = k/density
        
        # scale the existing set of probabilities like we did before to get a reduced lottery
        newprobabilities = [(1-prob) * a for a in probabilities]

        # if there weren't any, this is the first run, so we add a base probability
        if newprobabilities == []: newprobabilities.append(1-prob)
        # add the new probability
        newprobabilities.append(prob)

        # compute the next slider's position given the probability
        cx = wd - (1-prob)*(wd-prev[0])
        cy = hd - (1-prob)*(hd-prev[1])
        mid = [int(round(cx)), int(round(cy))]

        # compute expected utility
        expec = (1-prob)*probability + prob*adjoutcomes[level+1]

        # if this is the last level...
        if level == len(sliders)-1:
          # compute variance
          var = sum([p*(a-expec)**2 for p, a in zip(newprobabilities, adjoutcomes)])

          """
            Here risk aversion is defined as an affinity to pursue lotteries of higher variance.

            Risk-loving individuals would seek to maximize variance with less regard to expected utility.

            Risk-neutral individuals care about expected utility only, without caring about variance.

            Risk-averse individuals seek to minimize variance with less regard to expected utility.

            We therefore let risk aversion be a parameter that creates a linear combination of expected utility and variance to determine the final indifference fields.
          """

          # adjust for risk aversion
          aver = nmap(aversion, 115, 171, -1, 1)

          # if its negative, we are risk loving
          if aver < 0:
            aver = abs(aver)
            sign = -1
          else: sign = 1

          # convert to HSL
          h = nmap((1-aver)*expec-sign*aver*var, min(adjoutcomes), max(adjoutcomes), 0, 276)
          # convert to RGB
          col = colorsys.hls_to_rgb(h/360, 0.5, 1)
          col = [int(round(255*col[0])), int(round(255*col[1])), int(round(255*col[2])), 255]
          # if we saw this point before, make the color partly transparent so we can see what's underneath
          if base.get_at(mid) != (255, 255, 255):
            col[3] = 128
          pg.draw.circle(base, col, mid, 1) # draw the color

        # otherwise, we recursively check the next level
        else:
          shaderecursive(level+1, expec, mid, newprobabilities)

    # launch the recursive function
    shaderecursive(0, adjoutcomes[0], points[0], [])
  
  # compute expected value (not utility)
  evx = sum([p*a for p, a in zip(ps, outcomes)])

  # some metrics
  screen.blit(font3.render(f"Lottery: ({', '.join(str(round(p, 3)) for p in ps)}) over ({', '.join([str(o) for o in outcomes])}), u(x) = {utilfunc}, ex. value = {round(evx, 3)}, var = {round(sum([p*(a-evx)**2 for p, a in zip(ps, outcomes)]), 3)}", True, (0,0,0)), (0,62))

  # expected utility text
  output = " + ".join([f"{round(p, 3)}*{round(v, 3)}" for p, v in zip(ps, adjoutcomes)])

  # variance text
  output2 = " + ".join([f"{round(p, 3)}*({round(a, 3)}-{round(ev, 3)})^2" for p, a in zip(ps, adjoutcomes)])

  # engine to display the text onscreen properly without it going off
  res = f"Expected Utility: {output} = {round(ev, 3)}"
  res2 = f"Variance: {output2} = {round(variance, 3)}"

  # starter font
  font2 = pg.font.Font("OpenSansEmoji.ttf", 40)

  ind = 40    # current size
  bup = False # flag for if we need to Break-UP the text

  while font2.size(res)[0] > width or font2.size(res2)[0] > width:
    # loop downwards until we either hit size 14 or fit on the screen
    ind-=1
    if ind == 14:
      bup = True
      break

    # recompute font with decremented size
    font2 = pg.font.Font("OpenSansEmoji.ttf", ind)

  # offset holder for if text is multiline
  counter = 6*height//7

  # if we have to break the text in half...
  if bup:
    if font2.size(res)[0] > width: # if its the first one...
      # split in half, increase the offset counter, draw the second half only
      new = res[len(res)//2:]
      res = res[:len(res)//2]
      screen.blit(font2.render(new, True, (0,0,0)), (width//2 - font2.size(new)[0]//2, 6*height//7 + font2.size(res)[1]+2))
      counter += font2.size(res)[1]+2

    if font2.size(res2)[0] > width: # if its the second one...
      # do the same thing to the second one as the first
      new = res2[len(res2)//2:]
      res2 = res2[:len(res2)//2]
      screen.blit(font2.render(new, True, (0,0,0)), (width//2 - font2.size(new)[0]//2, counter + 2*font2.size(res)[1])) # this one has to be adjusted for offset if the first one was multiline
      
  # draw the first halves of each text entry, which might be everything if it wasn't split in half
  screen.blit(font2.render(res, True, (0,0,0)), (width//2 - font2.size(res)[0]//2, 6*height//7))
  screen.blit(font2.render(res2, True, (0,0,0)), (width//2 - font2.size(res2)[0]//2, counter + font2.size(res)[1])) # this one has to be adjusted for offset if the first was multiline

  if not ignore: # if we weren't told to not display this, display this
    pg.display.flip()

# render twice because the first one won't display anything due to the indifference field render
render()
render()

# PyGame keyboard entry setting for multikeypress handling
pg.key.set_repeat()

sliding = False # variable for if we are moving a slider

mx, my = pg.mouse.get_pos() # current position of mouse onscreen

# user input loop
while True:

  # get the keys and mouse position
  keys = pg.key.get_pressed()
  x, y = pg.mouse.get_pos()

  # if the mouse was moved and we are moving a slider...
  if sliding and (mx != x or my != y):
    # compute the first vertex of the line
    if index == 0: first = points[index]
    else: first = sliders[index-1]

    # compute the x coord of the second vertex of the line
    wd = points[index+1][0]

    # if the mouse is between the x coord of the first vertex and second, reposition the slider and redraw
    if x in range(first[0], wd+1):
      fraction = (wd-x)/(wd-first[0]) # get probability from ratio of line lengths
      probabilities[index] = 1 - fraction # we need 1-p due to offset
      render()

  # sequence of input checks that allow highlighted text buttons
  if intercept(x, y, 0, 80, 32) and cur == 0 and len(inputs[cur]): 
    if hovering != 1:
      hovering = 1
      render()
  elif intercept(x, y, 92, 80, 75) and cur == 1 and len(inputs[cur]):
    if hovering != 2:
      hovering = 2
      render()
  elif intercept(x, y, 217, 80, 48) and cur == 2 and len(inputs[cur]):
    if hovering != 3:
      hovering = 3
      render()

  # sequence of input checks that allow changing the aversion and density sliders if they changed
  elif intercept(x, y, 115, 45, 56, 20):
    if x != aversion:
      aversion = x
      changed = False
      render()
      render()
  elif intercept(x, y, 340, 45, 60, 20):
    if x != rdensity:
      rdensity = x
      changed = False
      render()
      render()
  
  # otherwise delete all highlights
  elif hovering != 0:
    hovering = 0
    render()
  
  # save the current mouse position
  mx, my = x, y

  # check for user input events
  for event in pg.event.get():
    if event.type == QUIT: 
      sys.exit() # quit if we need to, otherwise it crashes

    # if the mouse was clicked...
    elif event.type == MOUSEBUTTONDOWN:

      # sequence of input checks for textbox highlighting
      if intercept(x, y, 32, 80):
        cur = 0
      elif intercept(x, y, 157, 80):
        cur = 1
      elif intercept(x, y, 265, 80):
        cur = 2

      # input check for adding a new outcome
      elif intercept(x, y, 0, 80, 32) and cur == 0 and len(inputs[cur]):
        if "." not in inputs[cur]:
          # its an integer if its not a decmial
          outcomes.append(int(inputs[cur]))
        else: outcomes.append(float(inputs[cur])) # otherwise its dec
        probabilities.append(0) # give it zero probability to save old ones

        # recompute the position of vertices onscreen
        seg = len(outcomes)//2 + 1
        offset = (width/(seg+1))//2

        # wipe everything and re-render
        inputs[cur] = ""
        cur = -1
        changed = False
        index = -1
        sliding = False
        render()
        render()

      # input check for removing an outcome
      elif intercept(x, y, 92, 80, 75) and cur == 1 and len(inputs[cur]):
        # the same decimal check thing
        if "." not in inputs[cur]: check = int(inputs[cur])
        else: check = float(inputs[cur])

        # if we actually found it and have at least 3 outcomes left...
        # we must have at least two vertices onscreen or we get index errors
        if check in outcomes and len(outcomes) > 2:
          # find the index, delete the outcome, delete its probability, recompute everything
          ind = outcomes.index(check)
          if ind >= len(probabilities): ind = len(probabilities)-1
          del probabilities[ind]
          del outcomes[ind]
          seg = len(outcomes)//2 + 1
          offset = (width/(seg+1))//2

        # wipe data and re-render
        inputs[cur] = ""
        cur = -1
        index = -1
        sliding = False
        changed = False
        render()
        render()

      # input check for utility function
      elif intercept(x, y, 217, 80, 48) and cur == 2 and len(inputs[2]):
        try: 
          # check if its a valid utility function
          toput = inputs[2]
          cur = -1
          check = sympify(toput)
          [float(check.subs(symbols("x"), a)) for a in outcomes]
        except:
          # if it isn't send an error
          textbox(265, 80, utilfunc, -1, True)
          inputs[2] = utilfunc
          screen.blit(font3.render("ERROR: Invalid utility function.", True, (255,0,0)), (130, 100))
          pg.display.flip()
          time.sleep(2)
        else:
          # its valid so set it and recompute everything
          utilfunc = toput
          changed = False
          index = -1
          sliding = False
          render()
          render()

      else:
        # otherwise check if we moved a slider
        cur = -1

        # give some leeway to clicking the slider
        for dx in range(-10, 11):
          for dy in range(-10, 11):
            # if we found it and its not the one we're clicking...
            if [x+dx, y+dy] in sliders and index != sliders.index([x+dx, y+dy]):
              # make it the one we're clicking
              sliding = True
              a = sliders.copy()
              a.reverse()
              index = len(sliders) - 1 - a.index([x+dx, y+dy])
              break
          else: continue
          break
        else: 
          # if we never found one, we're not sliding
          sliding = False
          index = -1
      
      # rerender with the new highlighting because maybe we clicked something
      render()
          

    # case where we get keyboard input
    elif event.type == KEYDOWN and cur != -1:
      # backspace key
      if event.key == K_BACKSPACE:
        inputs[cur] = inputs[cur][:-1]
        render()

      # only allow number characters for number inputs, and also cap phrase lengths
      elif ((event.unicode in "1234567890" or (event.unicode == "." and "." not in inputs[cur]) or (event.unicode == "-" and inputs[cur] == "")) and len(inputs[cur]) < 6) or (cur == 2 and len(inputs[cur]) < 12):
        keys = pg.key.get_pressed()
        k = event.unicode
        # handle the shift button
        convert = dict(zip("`1234567890-=[]\\;',./", "~!@#$%^&*()_+{}|:\"<>?"))
        if keys[K_RSHIFT] or keys[K_LSHIFT]:
          k = k.upper()
          if k in convert:
            k = convert[k]

        # add the key and rerender with the phrase displayed in the text box
        inputs[cur] += k
        render()