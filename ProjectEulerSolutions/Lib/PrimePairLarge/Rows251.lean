import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows077

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7993_small : List Nat := PrimePairUpper.smallRow7993
theorem row7993_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7993 row7993_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7993_covers

def row7993_block_005 : List Nat := [9133, 9199, 9439, 10321]
def plan7993_block_005 : List Budget.Choice := [(.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 29), (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 31), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 31), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .second, (.firstFactor 29), (.secondFactor 29), .possible, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 37), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 29), (.firstFactor 17), .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 37), .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row7993_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7993 row7993_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7993_block_005
  · decide
  · rfl

def row7993_block_006 : List Nat := [10891, 11239]
def plan7993_block_006 : List Budget.Choice := [.first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 23), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .possible, .second, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .second, .first, .possible, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 7), .first, .second, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7993_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7993 row7993_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7993_block_006
  · decide
  · rfl

def row7993_block_007 : List Nat := [13003, 13441]
def plan7993_block_007 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 17), (.firstFactor 37), .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 37), .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .second, .first, (.secondFactor 23), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 19), .first, (.firstFactor 37), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .second, .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row7993_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7993 row7993_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7993_block_007
  · decide
  · rfl

def row7993_block_008 : List Nat := []
def plan7993_block_008 : List Budget.Choice := [(.secondFactor 17), .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 19), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .second, .second, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .second, (.firstFactor 31), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7993_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7993 row7993_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7993_block_008
  · decide
  · rfl

def row7993_block_009 : List Nat := [16831, 17209, 17317, 17851, 17881]
def plan7993_block_009 : List Budget.Choice := [(.secondFactor 17), .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .possible, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, (.firstFactor 11), .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .second, (.firstFactor 11), (.secondFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .possible, .second, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7993_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7993 row7993_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7993_block_009
  · decide
  · rfl

def row7993_block_010 : List Nat := []
theorem row7993_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7993 row7993_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7993_block_011 : List Nat := []
theorem row7993_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7993 row7993_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7993_block_012 : List Nat := []
theorem row7993_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7993 row7993_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7993_block_013 : List Nat := []
theorem row7993_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7993 row7993_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7993_large_left_left : List Nat := row7993_small ++ row7993_block_005
theorem row7993_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7993 row7993_large_left_left :=
  Budget.RowUpperBound.branch row7993_small_covers row7993_block_005_covers

def row7993_large_left_right_right : List Nat := row7993_block_007 ++ row7993_block_008
theorem row7993_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7993 row7993_large_left_right_right :=
  Budget.RowUpperBound.branch row7993_block_007_covers row7993_block_008_covers

def row7993_large_left_right : List Nat := row7993_block_006 ++ row7993_large_left_right_right
theorem row7993_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7993 row7993_large_left_right :=
  Budget.RowUpperBound.branch row7993_block_006_covers row7993_large_left_right_right_covers

def row7993_large_left : List Nat := row7993_large_left_left ++ row7993_large_left_right
theorem row7993_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7993 row7993_large_left :=
  Budget.RowUpperBound.branch row7993_large_left_left_covers row7993_large_left_right_covers

def row7993_large_right_left : List Nat := row7993_block_009 ++ row7993_block_010
theorem row7993_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7993 row7993_large_right_left :=
  Budget.RowUpperBound.branch row7993_block_009_covers row7993_block_010_covers

def row7993_large_right_right_right : List Nat := row7993_block_012 ++ row7993_block_013
theorem row7993_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7993 row7993_large_right_right_right :=
  Budget.RowUpperBound.branch row7993_block_012_covers row7993_block_013_covers

def row7993_large_right_right : List Nat := row7993_block_011 ++ row7993_large_right_right_right
theorem row7993_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7993 row7993_large_right_right :=
  Budget.RowUpperBound.branch row7993_block_011_covers row7993_large_right_right_right_covers

def row7993_large_right : List Nat := row7993_large_right_left ++ row7993_large_right_right
theorem row7993_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7993 row7993_large_right :=
  Budget.RowUpperBound.branch row7993_large_right_left_covers row7993_large_right_right_covers

def row7993_large : List Nat := row7993_large_left ++ row7993_large_right
theorem row7993_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7993 row7993_large :=
  Budget.RowUpperBound.branch row7993_large_left_covers row7993_large_right_covers

def row7993 : List Nat := row7993_large
theorem row7993_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7993 row7993 := row7993_large_covers

def row8009_small : List Nat := PrimePairUpper.smallRow8009
theorem row8009_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8009 row8009_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8009_covers

def row8009_block_005 : List Nat := [9311, 9419, 9491, 10193]
def plan8009_block_005 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 37), .first, .second, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 13), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 23), .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 37), .possible, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 29), .possible, (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 29), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .possible, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8009_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8009 row8009_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8009_block_005
  · decide
  · rfl

def row8009_block_006 : List Nat := [10739, 11393, 11621]
def plan8009_block_006 : List Budget.Choice := [(.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 29), .first, .possible, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .second, (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 29), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8009_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8009 row8009_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8009_block_006
  · decide
  · rfl

def row8009_block_007 : List Nat := [12821, 13187, 13649, 14249]
def plan8009_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .second, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 29), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 31), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8009_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8009 row8009_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8009_block_007
  · decide
  · rfl

def row8009_block_008 : List Nat := [14969, 15233, 15263, 15581, 15749, 16349]
def plan8009_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 13), .second, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 31), .first, .second, .first, .second, .first, .first, .first, (.firstFactor 19), .possible, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .possible, .second, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .second, .second, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 17), (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8009_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8009 row8009_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8009_block_008
  · decide
  · rfl

def row8009_block_009 : List Nat := [16619, 16661, 16703, 17123, 17189, 17483, 17921]
def plan8009_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), .possible, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 37), .first, .second, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .possible, (.secondFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8009_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8009 row8009_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8009_block_009
  · decide
  · rfl

def row8009_block_010 : List Nat := []
theorem row8009_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8009 row8009_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8009_block_011 : List Nat := []
theorem row8009_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8009 row8009_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8009_block_012 : List Nat := []
theorem row8009_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8009 row8009_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8009_block_013 : List Nat := []
theorem row8009_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8009 row8009_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8009_large_left_left : List Nat := row8009_small ++ row8009_block_005
theorem row8009_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8009 row8009_large_left_left :=
  Budget.RowUpperBound.branch row8009_small_covers row8009_block_005_covers

def row8009_large_left_right_right : List Nat := row8009_block_007 ++ row8009_block_008
theorem row8009_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8009 row8009_large_left_right_right :=
  Budget.RowUpperBound.branch row8009_block_007_covers row8009_block_008_covers

def row8009_large_left_right : List Nat := row8009_block_006 ++ row8009_large_left_right_right
theorem row8009_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8009 row8009_large_left_right :=
  Budget.RowUpperBound.branch row8009_block_006_covers row8009_large_left_right_right_covers

def row8009_large_left : List Nat := row8009_large_left_left ++ row8009_large_left_right
theorem row8009_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8009 row8009_large_left :=
  Budget.RowUpperBound.branch row8009_large_left_left_covers row8009_large_left_right_covers

def row8009_large_right_left : List Nat := row8009_block_009 ++ row8009_block_010
theorem row8009_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8009 row8009_large_right_left :=
  Budget.RowUpperBound.branch row8009_block_009_covers row8009_block_010_covers

def row8009_large_right_right_right : List Nat := row8009_block_012 ++ row8009_block_013
theorem row8009_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8009 row8009_large_right_right_right :=
  Budget.RowUpperBound.branch row8009_block_012_covers row8009_block_013_covers

def row8009_large_right_right : List Nat := row8009_block_011 ++ row8009_large_right_right_right
theorem row8009_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8009 row8009_large_right_right :=
  Budget.RowUpperBound.branch row8009_block_011_covers row8009_large_right_right_right_covers

def row8009_large_right : List Nat := row8009_large_right_left ++ row8009_large_right_right
theorem row8009_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8009 row8009_large_right :=
  Budget.RowUpperBound.branch row8009_large_right_left_covers row8009_large_right_right_covers

def row8009_large : List Nat := row8009_large_left ++ row8009_large_right
theorem row8009_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8009 row8009_large :=
  Budget.RowUpperBound.branch row8009_large_left_covers row8009_large_right_covers

def row8009 : List Nat := row8009_large
theorem row8009_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8009 row8009 := row8009_large_covers

def row8011_small : List Nat := PrimePairUpper.smallRow8011
theorem row8011_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8011 row8011_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8011_covers

def row8011_block_005 : List Nat := [8581, 8707, 8971, 9397, 9439]
def plan8011_block_005 : List Budget.Choice := [(.firstFactor 13), .second, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 29), (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .second, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .possible, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 23), (.firstFactor 23), (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row8011_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8011 row8011_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8011_block_005
  · decide
  · rfl

def row8011_block_006 : List Nat := [11743, 11821, 12373]
def plan8011_block_006 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 37), .first, .second, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.secondFactor 11), .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, (.firstFactor 23), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, .first, .possible, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row8011_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8011 row8011_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8011_block_006
  · decide
  · rfl

def row8011_block_007 : List Nat := [12457, 12619, 13711]
def plan8011_block_007 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 23), .second, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8011_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8011 row8011_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8011_block_007
  · decide
  · rfl

def row8011_block_008 : List Nat := [15451, 16411]
def plan8011_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 13), .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), .second, (.firstFactor 13), (.firstFactor 31), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 13), .possible, .first, .first, .first, .first, .second, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8011_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8011 row8011_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8011_block_008
  · decide
  · rfl

def row8011_block_009 : List Nat := [16831, 17167, 17467, 17659]
def plan8011_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 11), (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 29), .first, .second, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 7), (.firstFactor 11), .possible, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 23), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .second, .first, .first, .second, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8011_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8011 row8011_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8011_block_009
  · decide
  · rfl

def row8011_block_010 : List Nat := []
theorem row8011_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8011 row8011_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8011_block_011 : List Nat := []
theorem row8011_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8011 row8011_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8011_block_012 : List Nat := []
theorem row8011_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8011 row8011_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8011_block_013 : List Nat := []
theorem row8011_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8011 row8011_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8011_large_left_left : List Nat := row8011_small ++ row8011_block_005
theorem row8011_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8011 row8011_large_left_left :=
  Budget.RowUpperBound.branch row8011_small_covers row8011_block_005_covers

def row8011_large_left_right_right : List Nat := row8011_block_007 ++ row8011_block_008
theorem row8011_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8011 row8011_large_left_right_right :=
  Budget.RowUpperBound.branch row8011_block_007_covers row8011_block_008_covers

def row8011_large_left_right : List Nat := row8011_block_006 ++ row8011_large_left_right_right
theorem row8011_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8011 row8011_large_left_right :=
  Budget.RowUpperBound.branch row8011_block_006_covers row8011_large_left_right_right_covers

def row8011_large_left : List Nat := row8011_large_left_left ++ row8011_large_left_right
theorem row8011_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8011 row8011_large_left :=
  Budget.RowUpperBound.branch row8011_large_left_left_covers row8011_large_left_right_covers

def row8011_large_right_left : List Nat := row8011_block_009 ++ row8011_block_010
theorem row8011_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8011 row8011_large_right_left :=
  Budget.RowUpperBound.branch row8011_block_009_covers row8011_block_010_covers

def row8011_large_right_right_right : List Nat := row8011_block_012 ++ row8011_block_013
theorem row8011_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8011 row8011_large_right_right_right :=
  Budget.RowUpperBound.branch row8011_block_012_covers row8011_block_013_covers

def row8011_large_right_right : List Nat := row8011_block_011 ++ row8011_large_right_right_right
theorem row8011_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8011 row8011_large_right_right :=
  Budget.RowUpperBound.branch row8011_block_011_covers row8011_large_right_right_right_covers

def row8011_large_right : List Nat := row8011_large_right_left ++ row8011_large_right_right
theorem row8011_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8011 row8011_large_right :=
  Budget.RowUpperBound.branch row8011_large_right_left_covers row8011_large_right_right_covers

def row8011_large : List Nat := row8011_large_left ++ row8011_large_right
theorem row8011_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8011 row8011_large :=
  Budget.RowUpperBound.branch row8011_large_left_covers row8011_large_right_covers

def row8011 : List Nat := row8011_large
theorem row8011_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8011 row8011 := row8011_large_covers

def row8017_small : List Nat := PrimePairUpper.smallRow8017
theorem row8017_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8017 row8017_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8017_covers

def row8017_block_005 : List Nat := [8779, 9241, 9349, 9511, 9613, 10093]
def plan8017_block_005 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8017_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8017 row8017_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8017_block_005
  · decide
  · rfl

def row8017_block_006 : List Nat := [10627, 10753, 10957, 11059, 11527, 11839, 11971]
def plan8017_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .possible, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .possible, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 23), .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8017_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8017 row8017_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8017_block_006
  · decide
  · rfl

def row8017_block_007 : List Nat := [12517, 13267]
def plan8017_block_007 : List Budget.Choice := [(.firstFactor 31), (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .possible, .second, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .second, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 31), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second]
set_option maxRecDepth 2048 in
theorem row8017_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8017 row8017_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8017_block_007
  · decide
  · rfl

def row8017_block_008 : List Nat := [14503, 15307, 15973, 16417]
def plan8017_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .possible, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 37), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .second, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8017_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8017 row8017_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8017_block_008
  · decide
  · rfl

def row8017_block_009 : List Nat := [16963, 17827]
def plan8017_block_009 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 31), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.secondFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8017_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8017 row8017_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8017_block_009
  · decide
  · rfl

def row8017_block_010 : List Nat := []
theorem row8017_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8017 row8017_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8017_block_011 : List Nat := []
theorem row8017_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8017 row8017_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8017_block_012 : List Nat := []
theorem row8017_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8017 row8017_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8017_block_013 : List Nat := []
theorem row8017_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8017 row8017_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8017_large_left_left : List Nat := row8017_small ++ row8017_block_005
theorem row8017_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8017 row8017_large_left_left :=
  Budget.RowUpperBound.branch row8017_small_covers row8017_block_005_covers

def row8017_large_left_right_right : List Nat := row8017_block_007 ++ row8017_block_008
theorem row8017_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8017 row8017_large_left_right_right :=
  Budget.RowUpperBound.branch row8017_block_007_covers row8017_block_008_covers

def row8017_large_left_right : List Nat := row8017_block_006 ++ row8017_large_left_right_right
theorem row8017_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8017 row8017_large_left_right :=
  Budget.RowUpperBound.branch row8017_block_006_covers row8017_large_left_right_right_covers

def row8017_large_left : List Nat := row8017_large_left_left ++ row8017_large_left_right
theorem row8017_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8017 row8017_large_left :=
  Budget.RowUpperBound.branch row8017_large_left_left_covers row8017_large_left_right_covers

def row8017_large_right_left : List Nat := row8017_block_009 ++ row8017_block_010
theorem row8017_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8017 row8017_large_right_left :=
  Budget.RowUpperBound.branch row8017_block_009_covers row8017_block_010_covers

def row8017_large_right_right_right : List Nat := row8017_block_012 ++ row8017_block_013
theorem row8017_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8017 row8017_large_right_right_right :=
  Budget.RowUpperBound.branch row8017_block_012_covers row8017_block_013_covers

def row8017_large_right_right : List Nat := row8017_block_011 ++ row8017_large_right_right_right
theorem row8017_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8017 row8017_large_right_right :=
  Budget.RowUpperBound.branch row8017_block_011_covers row8017_large_right_right_right_covers

def row8017_large_right : List Nat := row8017_large_right_left ++ row8017_large_right_right
theorem row8017_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8017 row8017_large_right :=
  Budget.RowUpperBound.branch row8017_large_right_left_covers row8017_large_right_right_covers

def row8017_large : List Nat := row8017_large_left ++ row8017_large_right
theorem row8017_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8017 row8017_large :=
  Budget.RowUpperBound.branch row8017_large_left_covers row8017_large_right_covers

def row8017 : List Nat := row8017_large
theorem row8017_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8017 row8017 := row8017_large_covers

end ProjectEulerSolutions.PrimePairLarge
