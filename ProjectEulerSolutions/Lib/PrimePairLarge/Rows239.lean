import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows071

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7559_small : List Nat := PrimePairUpper.smallRow7559
theorem row7559_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7559 row7559_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7559_covers

def row7559_block_005 : List Nat := [8597, 8849, 9281]
def plan7559_block_005 : List Budget.Choice := [.first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .possible, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, .second, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 29), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 31), .first, (.firstFactor 17), .first, .second, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row7559_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7559 row7559_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7559_block_005
  · decide
  · rfl

def row7559_block_006 : List Nat := [10667, 11159]
def plan7559_block_006 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 23), (.secondFactor 23), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7559_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7559 row7559_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7559_block_006
  · decide
  · rfl

def row7559_block_007 : List Nat := [13523, 14057, 14153, 14321]
def plan7559_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 23), (.secondFactor 11), (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .second, .second, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 29), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 31), .first, .second, .first, (.firstFactor 23), .second, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row7559_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7559 row7559_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7559_block_007
  · decide
  · rfl

def row7559_block_008 : List Nat := [14657, 14813, 15527, 15581]
def plan7559_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 37), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), (.secondFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 29), (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 31), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 19), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 31), (.firstFactor 17), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row7559_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7559 row7559_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7559_block_008
  · decide
  · rfl

def row7559_block_009 : List Nat := [16979, 17783, 17837]
def plan7559_block_009 : List Budget.Choice := [.first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 31), .first, .first, .second, (.secondFactor 19), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 23), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 37), .first, .possible, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7559_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7559 row7559_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7559_block_009
  · decide
  · rfl

def row7559_block_010 : List Nat := []
theorem row7559_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7559 row7559_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7559_block_011 : List Nat := []
theorem row7559_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7559 row7559_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7559_block_012 : List Nat := []
theorem row7559_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7559 row7559_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7559_block_013 : List Nat := []
theorem row7559_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7559 row7559_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7559_large_left_left : List Nat := row7559_small ++ row7559_block_005
theorem row7559_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7559 row7559_large_left_left :=
  Budget.RowUpperBound.branch row7559_small_covers row7559_block_005_covers

def row7559_large_left_right_right : List Nat := row7559_block_007 ++ row7559_block_008
theorem row7559_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7559 row7559_large_left_right_right :=
  Budget.RowUpperBound.branch row7559_block_007_covers row7559_block_008_covers

def row7559_large_left_right : List Nat := row7559_block_006 ++ row7559_large_left_right_right
theorem row7559_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7559 row7559_large_left_right :=
  Budget.RowUpperBound.branch row7559_block_006_covers row7559_large_left_right_right_covers

def row7559_large_left : List Nat := row7559_large_left_left ++ row7559_large_left_right
theorem row7559_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7559 row7559_large_left :=
  Budget.RowUpperBound.branch row7559_large_left_left_covers row7559_large_left_right_covers

def row7559_large_right_left : List Nat := row7559_block_009 ++ row7559_block_010
theorem row7559_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7559 row7559_large_right_left :=
  Budget.RowUpperBound.branch row7559_block_009_covers row7559_block_010_covers

def row7559_large_right_right_right : List Nat := row7559_block_012 ++ row7559_block_013
theorem row7559_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7559 row7559_large_right_right_right :=
  Budget.RowUpperBound.branch row7559_block_012_covers row7559_block_013_covers

def row7559_large_right_right : List Nat := row7559_block_011 ++ row7559_large_right_right_right
theorem row7559_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7559 row7559_large_right_right :=
  Budget.RowUpperBound.branch row7559_block_011_covers row7559_large_right_right_right_covers

def row7559_large_right : List Nat := row7559_large_right_left ++ row7559_large_right_right
theorem row7559_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7559 row7559_large_right :=
  Budget.RowUpperBound.branch row7559_large_right_left_covers row7559_large_right_right_covers

def row7559_large : List Nat := row7559_large_left ++ row7559_large_right
theorem row7559_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7559 row7559_large :=
  Budget.RowUpperBound.branch row7559_large_left_covers row7559_large_right_covers

def row7559 : List Nat := row7559_large
theorem row7559_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7559 row7559 := row7559_large_covers

def row7561_small : List Nat := PrimePairUpper.smallRow7561
theorem row7561_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7561 row7561_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7561_covers

def row7561_block_005 : List Nat := [8821, 8887, 9127, 9337, 9433, 9463]
def plan7561_block_005 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 19), (.secondFactor 17), .first, .first, (.firstFactor 11), .second, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .second, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .possible, (.secondFactor 7), .first, .second, (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 17), .first, .first, (.secondFactor 29), .first, .possible, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .possible, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7561_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7561 row7561_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7561_block_005
  · decide
  · rfl

def row7561_block_006 : List Nat := [10891]
def plan7561_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 37), (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .second, (.firstFactor 13), .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .second, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row7561_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7561 row7561_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7561_block_006
  · decide
  · rfl

def row7561_block_007 : List Nat := [12919]
def plan7561_block_007 : List Budget.Choice := [(.secondFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .second, .first, .second, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .second, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row7561_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7561 row7561_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7561_block_007
  · decide
  · rfl

def row7561_block_008 : List Nat := [15667, 16069]
def plan7561_block_008 : List Budget.Choice := [.first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 37), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7561_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7561 row7561_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7561_block_008
  · decide
  · rfl

def row7561_block_009 : List Nat := [16519, 16657, 17599, 18229, 18379, 18397]
def plan7561_block_009 : List Budget.Choice := [.possible, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .second, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), (.secondFactor 11), (.secondFactor 17), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), .first, .second, (.firstFactor 7), (.secondFactor 13), .first, .first, .second, .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, (.secondFactor 17), (.firstFactor 19), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .possible, .possible, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7561_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7561 row7561_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7561_block_009
  · decide
  · rfl

def row7561_block_010 : List Nat := []
theorem row7561_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7561 row7561_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7561_block_011 : List Nat := []
theorem row7561_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7561 row7561_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7561_block_012 : List Nat := []
theorem row7561_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7561 row7561_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7561_block_013 : List Nat := []
theorem row7561_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7561 row7561_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7561_large_left_left : List Nat := row7561_small ++ row7561_block_005
theorem row7561_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7561 row7561_large_left_left :=
  Budget.RowUpperBound.branch row7561_small_covers row7561_block_005_covers

def row7561_large_left_right_right : List Nat := row7561_block_007 ++ row7561_block_008
theorem row7561_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7561 row7561_large_left_right_right :=
  Budget.RowUpperBound.branch row7561_block_007_covers row7561_block_008_covers

def row7561_large_left_right : List Nat := row7561_block_006 ++ row7561_large_left_right_right
theorem row7561_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7561 row7561_large_left_right :=
  Budget.RowUpperBound.branch row7561_block_006_covers row7561_large_left_right_right_covers

def row7561_large_left : List Nat := row7561_large_left_left ++ row7561_large_left_right
theorem row7561_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7561 row7561_large_left :=
  Budget.RowUpperBound.branch row7561_large_left_left_covers row7561_large_left_right_covers

def row7561_large_right_left : List Nat := row7561_block_009 ++ row7561_block_010
theorem row7561_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7561 row7561_large_right_left :=
  Budget.RowUpperBound.branch row7561_block_009_covers row7561_block_010_covers

def row7561_large_right_right_right : List Nat := row7561_block_012 ++ row7561_block_013
theorem row7561_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7561 row7561_large_right_right_right :=
  Budget.RowUpperBound.branch row7561_block_012_covers row7561_block_013_covers

def row7561_large_right_right : List Nat := row7561_block_011 ++ row7561_large_right_right_right
theorem row7561_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7561 row7561_large_right_right :=
  Budget.RowUpperBound.branch row7561_block_011_covers row7561_large_right_right_right_covers

def row7561_large_right : List Nat := row7561_large_right_left ++ row7561_large_right_right
theorem row7561_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7561 row7561_large_right :=
  Budget.RowUpperBound.branch row7561_large_right_left_covers row7561_large_right_right_covers

def row7561_large : List Nat := row7561_large_left ++ row7561_large_right
theorem row7561_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7561 row7561_large :=
  Budget.RowUpperBound.branch row7561_large_left_covers row7561_large_right_covers

def row7561 : List Nat := row7561_large
theorem row7561_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7561 row7561 := row7561_large_covers

def row7573_small : List Nat := PrimePairUpper.smallRow7573
theorem row7573_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7573 row7573_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7573_covers

def row7573_block_005 : List Nat := [8419, 8539, 9133, 9787]
def plan7573_block_005 : List Budget.Choice := [(.firstFactor 19), .possible, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 23), .first, .second, .first, .second, .first, .second, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 17), .possible, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .second, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 29), (.secondFactor 13), .first, .second, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7573_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7573 row7573_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7573_block_005
  · decide
  · rfl

def row7573_block_006 : List Nat := [10651, 10891, 11161, 11383, 11719, 12163]
def plan7573_block_006 : List Budget.Choice := [.first, .second, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .second, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, .possible, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .possible, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .second, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, .possible, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .second, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 31), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7573_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7573 row7573_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7573_block_006
  · decide
  · rfl

def row7573_block_007 : List Nat := [12979, 13537, 13633, 13831]
def plan7573_block_007 : List Budget.Choice := [(.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .second, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 37), .first, (.secondFactor 11), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .possible, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 29), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .possible, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 31)]
set_option maxRecDepth 2048 in
theorem row7573_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7573 row7573_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7573_block_007
  · decide
  · rfl

def row7573_block_008 : List Nat := [14563, 15217, 15739, 15991, 16141]
def plan7573_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 29), (.secondFactor 7), .possible, .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .possible, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, .possible, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7573_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7573 row7573_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7573_block_008
  · decide
  · rfl

def row7573_block_009 : List Nat := [17203, 18427]
def plan7573_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 23), .first, .second, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 23), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 23), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7573_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7573 row7573_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7573_block_009
  · decide
  · rfl

def row7573_block_010 : List Nat := []
theorem row7573_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7573 row7573_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7573_block_011 : List Nat := []
theorem row7573_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7573 row7573_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7573_block_012 : List Nat := []
theorem row7573_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7573 row7573_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7573_block_013 : List Nat := []
theorem row7573_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7573 row7573_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7573_large_left_left : List Nat := row7573_small ++ row7573_block_005
theorem row7573_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7573 row7573_large_left_left :=
  Budget.RowUpperBound.branch row7573_small_covers row7573_block_005_covers

def row7573_large_left_right_right : List Nat := row7573_block_007 ++ row7573_block_008
theorem row7573_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7573 row7573_large_left_right_right :=
  Budget.RowUpperBound.branch row7573_block_007_covers row7573_block_008_covers

def row7573_large_left_right : List Nat := row7573_block_006 ++ row7573_large_left_right_right
theorem row7573_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7573 row7573_large_left_right :=
  Budget.RowUpperBound.branch row7573_block_006_covers row7573_large_left_right_right_covers

def row7573_large_left : List Nat := row7573_large_left_left ++ row7573_large_left_right
theorem row7573_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7573 row7573_large_left :=
  Budget.RowUpperBound.branch row7573_large_left_left_covers row7573_large_left_right_covers

def row7573_large_right_left : List Nat := row7573_block_009 ++ row7573_block_010
theorem row7573_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7573 row7573_large_right_left :=
  Budget.RowUpperBound.branch row7573_block_009_covers row7573_block_010_covers

def row7573_large_right_right_right : List Nat := row7573_block_012 ++ row7573_block_013
theorem row7573_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7573 row7573_large_right_right_right :=
  Budget.RowUpperBound.branch row7573_block_012_covers row7573_block_013_covers

def row7573_large_right_right : List Nat := row7573_block_011 ++ row7573_large_right_right_right
theorem row7573_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7573 row7573_large_right_right :=
  Budget.RowUpperBound.branch row7573_block_011_covers row7573_large_right_right_right_covers

def row7573_large_right : List Nat := row7573_large_right_left ++ row7573_large_right_right
theorem row7573_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7573 row7573_large_right :=
  Budget.RowUpperBound.branch row7573_large_right_left_covers row7573_large_right_right_covers

def row7573_large : List Nat := row7573_large_left ++ row7573_large_right
theorem row7573_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7573 row7573_large :=
  Budget.RowUpperBound.branch row7573_large_left_covers row7573_large_right_covers

def row7573 : List Nat := row7573_large
theorem row7573_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7573 row7573 := row7573_large_covers

def row7577_small : List Nat := PrimePairUpper.smallRow7577
theorem row7577_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7577 row7577_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7577_covers

def row7577_block_005 : List Nat := [8543, 8609, 8693, 8699, 9413, 9461, 10259]
def plan7577_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 31), (.firstFactor 19), .first, (.firstFactor 11), .first, .possible, .possible, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 31), (.secondFactor 19), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 37), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row7577_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7577 row7577_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7577_block_005
  · decide
  · rfl

def row7577_block_006 : List Nat := [10433, 11177, 12119, 12227]
def plan7577_block_006 : List Budget.Choice := [.first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .second, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .second, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 23), (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), (.secondFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .second, .first, (.secondFactor 11), .possible, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 29), .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row7577_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7577 row7577_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7577_block_006
  · decide
  · rfl

def row7577_block_007 : List Nat := [12653, 12911, 14303, 14411]
def plan7577_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .possible, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, (.firstFactor 13), .first, (.firstFactor 11), .second, (.secondFactor 13), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7577_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7577 row7577_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7577_block_007
  · decide
  · rfl

def row7577_block_008 : List Nat := [14831, 15641, 15731, 15971, 16103, 16361]
def plan7577_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .second, .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 11), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 29), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7577_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7577 row7577_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7577_block_008
  · decide
  · rfl

def row7577_block_009 : List Nat := [16811, 17027, 17729]
def plan7577_block_009 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .possible, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7577_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7577 row7577_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7577_block_009
  · decide
  · rfl

def row7577_block_010 : List Nat := []
theorem row7577_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7577 row7577_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7577_block_011 : List Nat := []
theorem row7577_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7577 row7577_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7577_block_012 : List Nat := []
theorem row7577_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7577 row7577_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7577_block_013 : List Nat := []
theorem row7577_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7577 row7577_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7577_large_left_left : List Nat := row7577_small ++ row7577_block_005
theorem row7577_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7577 row7577_large_left_left :=
  Budget.RowUpperBound.branch row7577_small_covers row7577_block_005_covers

def row7577_large_left_right_right : List Nat := row7577_block_007 ++ row7577_block_008
theorem row7577_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7577 row7577_large_left_right_right :=
  Budget.RowUpperBound.branch row7577_block_007_covers row7577_block_008_covers

def row7577_large_left_right : List Nat := row7577_block_006 ++ row7577_large_left_right_right
theorem row7577_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7577 row7577_large_left_right :=
  Budget.RowUpperBound.branch row7577_block_006_covers row7577_large_left_right_right_covers

def row7577_large_left : List Nat := row7577_large_left_left ++ row7577_large_left_right
theorem row7577_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7577 row7577_large_left :=
  Budget.RowUpperBound.branch row7577_large_left_left_covers row7577_large_left_right_covers

def row7577_large_right_left : List Nat := row7577_block_009 ++ row7577_block_010
theorem row7577_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7577 row7577_large_right_left :=
  Budget.RowUpperBound.branch row7577_block_009_covers row7577_block_010_covers

def row7577_large_right_right_right : List Nat := row7577_block_012 ++ row7577_block_013
theorem row7577_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7577 row7577_large_right_right_right :=
  Budget.RowUpperBound.branch row7577_block_012_covers row7577_block_013_covers

def row7577_large_right_right : List Nat := row7577_block_011 ++ row7577_large_right_right_right
theorem row7577_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7577 row7577_large_right_right :=
  Budget.RowUpperBound.branch row7577_block_011_covers row7577_large_right_right_right_covers

def row7577_large_right : List Nat := row7577_large_right_left ++ row7577_large_right_right
theorem row7577_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7577 row7577_large_right :=
  Budget.RowUpperBound.branch row7577_large_right_left_covers row7577_large_right_right_covers

def row7577_large : List Nat := row7577_large_left ++ row7577_large_right
theorem row7577_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7577 row7577_large :=
  Budget.RowUpperBound.branch row7577_large_left_covers row7577_large_right_covers

def row7577 : List Nat := row7577_large
theorem row7577_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7577 row7577 := row7577_large_covers

end ProjectEulerSolutions.PrimePairLarge
