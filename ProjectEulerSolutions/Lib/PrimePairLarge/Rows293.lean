import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9479_small : List Nat := []
theorem row9479_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9479 row9479_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9479_block_005 : List Nat := [9533, 9941, 10163]
def plan9479_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 29), .possible, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row9479_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9479 row9479_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9479_block_005
  · decide
  · rfl

def row9479_block_006 : List Nat := [10433, 11057, 11411, 11969, 12281]
def plan9479_block_006 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .second, (.secondFactor 7), .second, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, .possible, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9479_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9479 row9479_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9479_block_006
  · decide
  · rfl

def row9479_block_007 : List Nat := [14153]
def plan9479_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .second, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 31), .first, .second, .first, .first, (.firstFactor 29), .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row9479_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9479 row9479_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9479_block_007
  · decide
  · rfl

def row9479_block_008 : List Nat := [14519, 14549, 15473, 15641, 16451]
def plan9479_block_008 : List Budget.Choice := [.first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 17), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 11), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), (.secondFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .possible, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 23), .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .second, (.secondFactor 7), (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 17), (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9479_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9479 row9479_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9479_block_008
  · decide
  · rfl

def row9479_block_009 : List Nat := []
def plan9479_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9479_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9479 row9479_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9479_block_009
  · decide
  · rfl

def row9479_block_010 : List Nat := []
theorem row9479_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9479 row9479_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9479_block_011 : List Nat := []
theorem row9479_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9479 row9479_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9479_block_012 : List Nat := []
theorem row9479_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9479 row9479_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9479_block_013 : List Nat := []
theorem row9479_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9479 row9479_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9479_large_left_left : List Nat := row9479_small ++ row9479_block_005
theorem row9479_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9479 row9479_large_left_left :=
  Budget.RowUpperBound.branch row9479_small_covers row9479_block_005_covers

def row9479_large_left_right_right : List Nat := row9479_block_007 ++ row9479_block_008
theorem row9479_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9479 row9479_large_left_right_right :=
  Budget.RowUpperBound.branch row9479_block_007_covers row9479_block_008_covers

def row9479_large_left_right : List Nat := row9479_block_006 ++ row9479_large_left_right_right
theorem row9479_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9479 row9479_large_left_right :=
  Budget.RowUpperBound.branch row9479_block_006_covers row9479_large_left_right_right_covers

def row9479_large_left : List Nat := row9479_large_left_left ++ row9479_large_left_right
theorem row9479_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9479 row9479_large_left :=
  Budget.RowUpperBound.branch row9479_large_left_left_covers row9479_large_left_right_covers

def row9479_large_right_left : List Nat := row9479_block_009 ++ row9479_block_010
theorem row9479_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9479 row9479_large_right_left :=
  Budget.RowUpperBound.branch row9479_block_009_covers row9479_block_010_covers

def row9479_large_right_right_right : List Nat := row9479_block_012 ++ row9479_block_013
theorem row9479_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9479 row9479_large_right_right_right :=
  Budget.RowUpperBound.branch row9479_block_012_covers row9479_block_013_covers

def row9479_large_right_right : List Nat := row9479_block_011 ++ row9479_large_right_right_right
theorem row9479_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9479 row9479_large_right_right :=
  Budget.RowUpperBound.branch row9479_block_011_covers row9479_large_right_right_right_covers

def row9479_large_right : List Nat := row9479_large_right_left ++ row9479_large_right_right
theorem row9479_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9479 row9479_large_right :=
  Budget.RowUpperBound.branch row9479_large_right_left_covers row9479_large_right_right_covers

def row9479_large : List Nat := row9479_large_left ++ row9479_large_right
theorem row9479_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9479 row9479_large :=
  Budget.RowUpperBound.branch row9479_large_left_covers row9479_large_right_covers

def row9479 : List Nat := row9479_large
theorem row9479_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9479 row9479 := row9479_large_covers

def row9491_small : List Nat := []
theorem row9491_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9491 row9491_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9491_block_005 : List Nat := [9521, 9623, 10181]
def plan9491_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 19), .first, .possible, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 23), .first, .second, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row9491_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9491 row9491_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9491_block_005
  · decide
  · rfl

def row9491_block_006 : List Nat := [11369]
def plan9491_block_006 : List Budget.Choice := [(.firstFactor 17), .first, .second, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .possible, .first, (.secondFactor 17), (.firstFactor 23), (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .second, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9491_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9491 row9491_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9491_block_006
  · decide
  · rfl

def row9491_block_007 : List Nat := [12611]
def plan9491_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 17), (.secondFactor 31), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .second, .first, .first, .first, .first, .second, .first, (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 11), .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9491_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9491 row9491_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9491_block_007
  · decide
  · rfl

def row9491_block_008 : List Nat := [14897, 16451]
def plan9491_block_008 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .second, (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 29), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 23), .first, (.firstFactor 19), .second, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 37), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .possible, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9491_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9491 row9491_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9491_block_008
  · decide
  · rfl

def row9491_block_009 : List Nat := []
def plan9491_block_009 : List Budget.Choice := [.first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9491_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9491 row9491_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9491_block_009
  · decide
  · rfl

def row9491_block_010 : List Nat := []
theorem row9491_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9491 row9491_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9491_block_011 : List Nat := []
theorem row9491_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9491 row9491_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9491_block_012 : List Nat := []
theorem row9491_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9491 row9491_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9491_block_013 : List Nat := []
theorem row9491_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9491 row9491_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9491_large_left_left : List Nat := row9491_small ++ row9491_block_005
theorem row9491_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9491 row9491_large_left_left :=
  Budget.RowUpperBound.branch row9491_small_covers row9491_block_005_covers

def row9491_large_left_right_right : List Nat := row9491_block_007 ++ row9491_block_008
theorem row9491_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9491 row9491_large_left_right_right :=
  Budget.RowUpperBound.branch row9491_block_007_covers row9491_block_008_covers

def row9491_large_left_right : List Nat := row9491_block_006 ++ row9491_large_left_right_right
theorem row9491_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9491 row9491_large_left_right :=
  Budget.RowUpperBound.branch row9491_block_006_covers row9491_large_left_right_right_covers

def row9491_large_left : List Nat := row9491_large_left_left ++ row9491_large_left_right
theorem row9491_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9491 row9491_large_left :=
  Budget.RowUpperBound.branch row9491_large_left_left_covers row9491_large_left_right_covers

def row9491_large_right_left : List Nat := row9491_block_009 ++ row9491_block_010
theorem row9491_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9491 row9491_large_right_left :=
  Budget.RowUpperBound.branch row9491_block_009_covers row9491_block_010_covers

def row9491_large_right_right_right : List Nat := row9491_block_012 ++ row9491_block_013
theorem row9491_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9491 row9491_large_right_right_right :=
  Budget.RowUpperBound.branch row9491_block_012_covers row9491_block_013_covers

def row9491_large_right_right : List Nat := row9491_block_011 ++ row9491_large_right_right_right
theorem row9491_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9491 row9491_large_right_right :=
  Budget.RowUpperBound.branch row9491_block_011_covers row9491_large_right_right_right_covers

def row9491_large_right : List Nat := row9491_large_right_left ++ row9491_large_right_right
theorem row9491_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9491 row9491_large_right :=
  Budget.RowUpperBound.branch row9491_large_right_left_covers row9491_large_right_right_covers

def row9491_large : List Nat := row9491_large_left ++ row9491_large_right
theorem row9491_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9491 row9491_large :=
  Budget.RowUpperBound.branch row9491_large_left_covers row9491_large_right_covers

def row9491 : List Nat := row9491_large
theorem row9491_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9491 row9491 := row9491_large_covers

def row9497_small : List Nat := []
theorem row9497_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9497 row9497_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9497_block_005 : List Nat := []
def plan9497_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, (.secondFactor 31), .first, (.secondFactor 13), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9497_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9497 row9497_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9497_block_005
  · decide
  · rfl

def row9497_block_006 : List Nat := [12161]
def plan9497_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 23), .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .second, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9497_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9497 row9497_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9497_block_006
  · decide
  · rfl

def row9497_block_007 : List Nat := [13421, 13883, 14177]
def plan9497_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 29), .first, .first, (.firstFactor 23), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .second, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 29), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .second, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9497_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9497 row9497_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9497_block_007
  · decide
  · rfl

def row9497_block_008 : List Nat := [14657, 15401, 15731, 16301]
def plan9497_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 23), .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 11), .first, .second, (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .possible, .second, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9497_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9497 row9497_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9497_block_008
  · decide
  · rfl

def row9497_block_009 : List Nat := []
def plan9497_block_009 : List Budget.Choice := [.first, (.firstFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9497_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9497 row9497_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9497_block_009
  · decide
  · rfl

def row9497_block_010 : List Nat := []
theorem row9497_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9497 row9497_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9497_block_011 : List Nat := []
theorem row9497_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9497 row9497_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9497_block_012 : List Nat := []
theorem row9497_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9497 row9497_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9497_block_013 : List Nat := []
theorem row9497_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9497 row9497_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9497_large_left_left : List Nat := row9497_small ++ row9497_block_005
theorem row9497_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9497 row9497_large_left_left :=
  Budget.RowUpperBound.branch row9497_small_covers row9497_block_005_covers

def row9497_large_left_right_right : List Nat := row9497_block_007 ++ row9497_block_008
theorem row9497_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9497 row9497_large_left_right_right :=
  Budget.RowUpperBound.branch row9497_block_007_covers row9497_block_008_covers

def row9497_large_left_right : List Nat := row9497_block_006 ++ row9497_large_left_right_right
theorem row9497_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9497 row9497_large_left_right :=
  Budget.RowUpperBound.branch row9497_block_006_covers row9497_large_left_right_right_covers

def row9497_large_left : List Nat := row9497_large_left_left ++ row9497_large_left_right
theorem row9497_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9497 row9497_large_left :=
  Budget.RowUpperBound.branch row9497_large_left_left_covers row9497_large_left_right_covers

def row9497_large_right_left : List Nat := row9497_block_009 ++ row9497_block_010
theorem row9497_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9497 row9497_large_right_left :=
  Budget.RowUpperBound.branch row9497_block_009_covers row9497_block_010_covers

def row9497_large_right_right_right : List Nat := row9497_block_012 ++ row9497_block_013
theorem row9497_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9497 row9497_large_right_right_right :=
  Budget.RowUpperBound.branch row9497_block_012_covers row9497_block_013_covers

def row9497_large_right_right : List Nat := row9497_block_011 ++ row9497_large_right_right_right
theorem row9497_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9497 row9497_large_right_right :=
  Budget.RowUpperBound.branch row9497_block_011_covers row9497_large_right_right_right_covers

def row9497_large_right : List Nat := row9497_large_right_left ++ row9497_large_right_right
theorem row9497_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9497 row9497_large_right :=
  Budget.RowUpperBound.branch row9497_large_right_left_covers row9497_large_right_right_covers

def row9497_large : List Nat := row9497_large_left ++ row9497_large_right
theorem row9497_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9497 row9497_large :=
  Budget.RowUpperBound.branch row9497_large_left_covers row9497_large_right_covers

def row9497 : List Nat := row9497_large
theorem row9497_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9497 row9497 := row9497_large_covers

def row9511_small : List Nat := []
theorem row9511_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9511 row9511_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9511_block_005 : List Nat := [10177]
def plan9511_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 19), .second, (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 37), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9511_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9511 row9511_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9511_block_005
  · decide
  · rfl

def row9511_block_006 : List Nat := [10837, 11311, 11383]
def plan9511_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .second, .first, .first, (.secondFactor 19), .second, (.secondFactor 11), .first, .first, (.secondFactor 29), (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 11), .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .second, .possible, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9511_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9511 row9511_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9511_block_006
  · decide
  · rfl

def row9511_block_007 : List Nat := [13183, 13879]
def plan9511_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), (.firstFactor 7), .second, .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9511_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9511 row9511_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9511_block_007
  · decide
  · rfl

def row9511_block_008 : List Nat := [14713, 14947, 15427, 15511, 16333, 16417]
def plan9511_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .possible, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, (.firstFactor 13), .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .second, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .possible, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.secondFactor 11), .first, .first, .possible, (.firstFactor 37), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9511_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9511 row9511_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9511_block_008
  · decide
  · rfl

def row9511_block_009 : List Nat := []
def plan9511_block_009 : List Budget.Choice := [.first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9511_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9511 row9511_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9511_block_009
  · decide
  · rfl

def row9511_block_010 : List Nat := []
theorem row9511_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9511 row9511_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9511_block_011 : List Nat := []
theorem row9511_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9511 row9511_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9511_block_012 : List Nat := []
theorem row9511_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9511 row9511_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9511_block_013 : List Nat := []
theorem row9511_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9511 row9511_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9511_large_left_left : List Nat := row9511_small ++ row9511_block_005
theorem row9511_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9511 row9511_large_left_left :=
  Budget.RowUpperBound.branch row9511_small_covers row9511_block_005_covers

def row9511_large_left_right_right : List Nat := row9511_block_007 ++ row9511_block_008
theorem row9511_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9511 row9511_large_left_right_right :=
  Budget.RowUpperBound.branch row9511_block_007_covers row9511_block_008_covers

def row9511_large_left_right : List Nat := row9511_block_006 ++ row9511_large_left_right_right
theorem row9511_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9511 row9511_large_left_right :=
  Budget.RowUpperBound.branch row9511_block_006_covers row9511_large_left_right_right_covers

def row9511_large_left : List Nat := row9511_large_left_left ++ row9511_large_left_right
theorem row9511_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9511 row9511_large_left :=
  Budget.RowUpperBound.branch row9511_large_left_left_covers row9511_large_left_right_covers

def row9511_large_right_left : List Nat := row9511_block_009 ++ row9511_block_010
theorem row9511_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9511 row9511_large_right_left :=
  Budget.RowUpperBound.branch row9511_block_009_covers row9511_block_010_covers

def row9511_large_right_right_right : List Nat := row9511_block_012 ++ row9511_block_013
theorem row9511_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9511 row9511_large_right_right_right :=
  Budget.RowUpperBound.branch row9511_block_012_covers row9511_block_013_covers

def row9511_large_right_right : List Nat := row9511_block_011 ++ row9511_large_right_right_right
theorem row9511_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9511 row9511_large_right_right :=
  Budget.RowUpperBound.branch row9511_block_011_covers row9511_large_right_right_right_covers

def row9511_large_right : List Nat := row9511_large_right_left ++ row9511_large_right_right
theorem row9511_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9511 row9511_large_right :=
  Budget.RowUpperBound.branch row9511_large_right_left_covers row9511_large_right_right_covers

def row9511_large : List Nat := row9511_large_left ++ row9511_large_right
theorem row9511_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9511 row9511_large :=
  Budget.RowUpperBound.branch row9511_large_left_covers row9511_large_right_covers

def row9511 : List Nat := row9511_large
theorem row9511_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9511 row9511 := row9511_large_covers

end ProjectEulerSolutions.PrimePairLarge
