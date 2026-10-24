import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows075

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7793_small : List Nat := PrimePairUpper.smallRow7793
theorem row7793_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7793 row7793_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7793_covers

def row7793_block_005 : List Nat := [8807, 9161, 9461, 9923]
def plan7793_block_005 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .possible, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), (.secondFactor 29), .first, .first, .first, (.secondFactor 31), .first, .first, .second, .first, .first, .first, .possible, .first, (.secondFactor 7), .second, (.firstFactor 23), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 13), (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 19), (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 11), .second, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7793_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7793 row7793_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7793_block_005
  · decide
  · rfl

def row7793_block_006 : List Nat := [10607, 10667, 11003, 11213, 11411, 11681, 12143, 12269, 12329, 12377]
def plan7793_block_006 : List Budget.Choice := [(.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .possible, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .possible, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 19), .first, .possible, .first, (.secondFactor 11), .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 13), .second, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .second, .possible, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 23), (.secondFactor 23), .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row7793_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7793 row7793_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7793_block_006
  · decide
  · rfl

def row7793_block_007 : List Nat := [12539, 12917, 12983]
def plan7793_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .possible, .first, .second, (.firstFactor 11), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .second, (.firstFactor 37), .first, .second, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 19), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .second, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 37), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7793_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7793 row7793_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7793_block_007
  · decide
  · rfl

def row7793_block_008 : List Nat := [14741, 15161, 16301, 16361]
def plan7793_block_008 : List Budget.Choice := [.first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .second, .first, (.secondFactor 23), (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .possible, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 23), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .possible, (.secondFactor 11), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7793_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7793 row7793_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7793_block_008
  · decide
  · rfl

def row7793_block_009 : List Nat := [16871, 17159, 17807]
def plan7793_block_009 : List Budget.Choice := [.first, .first, (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .possible, .first, .second, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .second, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, .possible, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7793_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7793 row7793_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7793_block_009
  · decide
  · rfl

def row7793_block_010 : List Nat := []
theorem row7793_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7793 row7793_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7793_block_011 : List Nat := []
theorem row7793_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7793 row7793_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7793_block_012 : List Nat := []
theorem row7793_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7793 row7793_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7793_block_013 : List Nat := []
theorem row7793_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7793 row7793_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7793_large_left_left : List Nat := row7793_small ++ row7793_block_005
theorem row7793_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7793 row7793_large_left_left :=
  Budget.RowUpperBound.branch row7793_small_covers row7793_block_005_covers

def row7793_large_left_right_right : List Nat := row7793_block_007 ++ row7793_block_008
theorem row7793_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7793 row7793_large_left_right_right :=
  Budget.RowUpperBound.branch row7793_block_007_covers row7793_block_008_covers

def row7793_large_left_right : List Nat := row7793_block_006 ++ row7793_large_left_right_right
theorem row7793_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7793 row7793_large_left_right :=
  Budget.RowUpperBound.branch row7793_block_006_covers row7793_large_left_right_right_covers

def row7793_large_left : List Nat := row7793_large_left_left ++ row7793_large_left_right
theorem row7793_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7793 row7793_large_left :=
  Budget.RowUpperBound.branch row7793_large_left_left_covers row7793_large_left_right_covers

def row7793_large_right_left : List Nat := row7793_block_009 ++ row7793_block_010
theorem row7793_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7793 row7793_large_right_left :=
  Budget.RowUpperBound.branch row7793_block_009_covers row7793_block_010_covers

def row7793_large_right_right_right : List Nat := row7793_block_012 ++ row7793_block_013
theorem row7793_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7793 row7793_large_right_right_right :=
  Budget.RowUpperBound.branch row7793_block_012_covers row7793_block_013_covers

def row7793_large_right_right : List Nat := row7793_block_011 ++ row7793_large_right_right_right
theorem row7793_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7793 row7793_large_right_right :=
  Budget.RowUpperBound.branch row7793_block_011_covers row7793_large_right_right_right_covers

def row7793_large_right : List Nat := row7793_large_right_left ++ row7793_large_right_right
theorem row7793_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7793 row7793_large_right :=
  Budget.RowUpperBound.branch row7793_large_right_left_covers row7793_large_right_right_covers

def row7793_large : List Nat := row7793_large_left ++ row7793_large_right
theorem row7793_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7793 row7793_large :=
  Budget.RowUpperBound.branch row7793_large_left_covers row7793_large_right_covers

def row7793 : List Nat := row7793_large
theorem row7793_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7793 row7793 := row7793_large_covers

def row7817_small : List Nat := PrimePairUpper.smallRow7817
theorem row7817_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7817 row7817_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7817_covers

def row7817_block_005 : List Nat := [8861, 9371, 9749, 9833]
def plan7817_block_005 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .second, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .possible, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 13), .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7817_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7817 row7817_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7817_block_005
  · decide
  · rfl

def row7817_block_006 : List Nat := [10859, 11369]
def plan7817_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 7), .possible, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .second, (.firstFactor 13), (.secondFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7817_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7817 row7817_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7817_block_006
  · decide
  · rfl

def row7817_block_007 : List Nat := [12641, 12713, 12791, 12923, 14411]
def plan7817_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .possible, (.firstFactor 31), (.firstFactor 37), (.firstFactor 7), .second, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .possible, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .second, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .second, (.secondFactor 11), (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row7817_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7817 row7817_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7817_block_007
  · decide
  · rfl

def row7817_block_008 : List Nat := [14657, 14831, 15227, 15731, 15923]
def plan7817_block_008 : List Budget.Choice := [.first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 29), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 17), (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .second, .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 37), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row7817_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7817 row7817_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7817_block_008
  · decide
  · rfl

def row7817_block_009 : List Nat := [16673, 17393]
def plan7817_block_009 : List Budget.Choice := [.first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 7), (.secondFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7817_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7817 row7817_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7817_block_009
  · decide
  · rfl

def row7817_block_010 : List Nat := []
theorem row7817_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7817 row7817_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7817_block_011 : List Nat := []
theorem row7817_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7817 row7817_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7817_block_012 : List Nat := []
theorem row7817_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7817 row7817_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7817_block_013 : List Nat := []
theorem row7817_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7817 row7817_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7817_large_left_left : List Nat := row7817_small ++ row7817_block_005
theorem row7817_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7817 row7817_large_left_left :=
  Budget.RowUpperBound.branch row7817_small_covers row7817_block_005_covers

def row7817_large_left_right_right : List Nat := row7817_block_007 ++ row7817_block_008
theorem row7817_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7817 row7817_large_left_right_right :=
  Budget.RowUpperBound.branch row7817_block_007_covers row7817_block_008_covers

def row7817_large_left_right : List Nat := row7817_block_006 ++ row7817_large_left_right_right
theorem row7817_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7817 row7817_large_left_right :=
  Budget.RowUpperBound.branch row7817_block_006_covers row7817_large_left_right_right_covers

def row7817_large_left : List Nat := row7817_large_left_left ++ row7817_large_left_right
theorem row7817_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7817 row7817_large_left :=
  Budget.RowUpperBound.branch row7817_large_left_left_covers row7817_large_left_right_covers

def row7817_large_right_left : List Nat := row7817_block_009 ++ row7817_block_010
theorem row7817_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7817 row7817_large_right_left :=
  Budget.RowUpperBound.branch row7817_block_009_covers row7817_block_010_covers

def row7817_large_right_right_right : List Nat := row7817_block_012 ++ row7817_block_013
theorem row7817_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7817 row7817_large_right_right_right :=
  Budget.RowUpperBound.branch row7817_block_012_covers row7817_block_013_covers

def row7817_large_right_right : List Nat := row7817_block_011 ++ row7817_large_right_right_right
theorem row7817_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7817 row7817_large_right_right :=
  Budget.RowUpperBound.branch row7817_block_011_covers row7817_large_right_right_right_covers

def row7817_large_right : List Nat := row7817_large_right_left ++ row7817_large_right_right
theorem row7817_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7817 row7817_large_right :=
  Budget.RowUpperBound.branch row7817_large_right_left_covers row7817_large_right_right_covers

def row7817_large : List Nat := row7817_large_left ++ row7817_large_right
theorem row7817_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7817 row7817_large :=
  Budget.RowUpperBound.branch row7817_large_left_covers row7817_large_right_covers

def row7817 : List Nat := row7817_large
theorem row7817_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7817 row7817 := row7817_large_covers

def row7823_small : List Nat := PrimePairUpper.smallRow7823
theorem row7823_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7823 row7823_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7823_covers

def row7823_block_005 : List Nat := [10007]
def plan7823_block_005 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .second, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .second, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 29), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7823_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7823 row7823_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7823_block_005
  · decide
  · rfl

def row7823_block_006 : List Nat := [10733, 11807, 12281]
def plan7823_block_006 : List Budget.Choice := [.second, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 31), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 31), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .possible, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, .possible, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row7823_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7823 row7823_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7823_block_006
  · decide
  · rfl

def row7823_block_007 : List Nat := [12437, 12917, 12959, 13151, 13457]
def plan7823_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .second, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .second, (.secondFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .second, .possible, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .possible, .first, .first, .first, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, (.secondFactor 11), .first, .first, .possible, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .possible, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 29), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 29), .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7823_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7823 row7823_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7823_block_007
  · decide
  · rfl

def row7823_block_008 : List Nat := [15467, 15551, 15923, 15971, 16103]
def plan7823_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 31), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .second, .first, .first, (.firstFactor 13), .second, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37), .second, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .second, .possible, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 23), .second, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .possible, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 19), .possible, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 17), .second, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7823_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7823 row7823_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7823_block_008
  · decide
  · rfl

def row7823_block_009 : List Nat := [17231, 17903]
def plan7823_block_009 : List Budget.Choice := [.first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .second, .second, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .possible, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 31), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .second, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7823_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7823 row7823_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7823_block_009
  · decide
  · rfl

def row7823_block_010 : List Nat := []
theorem row7823_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7823 row7823_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7823_block_011 : List Nat := []
theorem row7823_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7823 row7823_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7823_block_012 : List Nat := []
theorem row7823_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7823 row7823_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7823_block_013 : List Nat := []
theorem row7823_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7823 row7823_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7823_large_left_left : List Nat := row7823_small ++ row7823_block_005
theorem row7823_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7823 row7823_large_left_left :=
  Budget.RowUpperBound.branch row7823_small_covers row7823_block_005_covers

def row7823_large_left_right_right : List Nat := row7823_block_007 ++ row7823_block_008
theorem row7823_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7823 row7823_large_left_right_right :=
  Budget.RowUpperBound.branch row7823_block_007_covers row7823_block_008_covers

def row7823_large_left_right : List Nat := row7823_block_006 ++ row7823_large_left_right_right
theorem row7823_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7823 row7823_large_left_right :=
  Budget.RowUpperBound.branch row7823_block_006_covers row7823_large_left_right_right_covers

def row7823_large_left : List Nat := row7823_large_left_left ++ row7823_large_left_right
theorem row7823_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7823 row7823_large_left :=
  Budget.RowUpperBound.branch row7823_large_left_left_covers row7823_large_left_right_covers

def row7823_large_right_left : List Nat := row7823_block_009 ++ row7823_block_010
theorem row7823_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7823 row7823_large_right_left :=
  Budget.RowUpperBound.branch row7823_block_009_covers row7823_block_010_covers

def row7823_large_right_right_right : List Nat := row7823_block_012 ++ row7823_block_013
theorem row7823_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7823 row7823_large_right_right_right :=
  Budget.RowUpperBound.branch row7823_block_012_covers row7823_block_013_covers

def row7823_large_right_right : List Nat := row7823_block_011 ++ row7823_large_right_right_right
theorem row7823_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7823 row7823_large_right_right :=
  Budget.RowUpperBound.branch row7823_block_011_covers row7823_large_right_right_right_covers

def row7823_large_right : List Nat := row7823_large_right_left ++ row7823_large_right_right
theorem row7823_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7823 row7823_large_right :=
  Budget.RowUpperBound.branch row7823_large_right_left_covers row7823_large_right_right_covers

def row7823_large : List Nat := row7823_large_left ++ row7823_large_right
theorem row7823_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7823 row7823_large :=
  Budget.RowUpperBound.branch row7823_large_left_covers row7823_large_right_covers

def row7823 : List Nat := row7823_large
theorem row7823_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7823 row7823 := row7823_large_covers

def row7829_small : List Nat := PrimePairUpper.smallRow7829
theorem row7829_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7829 row7829_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7829_covers

def row7829_block_005 : List Nat := [8837]
def plan7829_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .second, .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 23), (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), .second, .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 11), (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row7829_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7829 row7829_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7829_block_005
  · decide
  · rfl

def row7829_block_006 : List Nat := [11471, 12071, 12107, 12227]
def plan7829_block_006 : List Budget.Choice := [.second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 17), .possible, .first, .first, (.firstFactor 11), (.firstFactor 13), .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .possible, (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row7829_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7829 row7829_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7829_block_006
  · decide
  · rfl

def row7829_block_007 : List Nat := [14423]
def plan7829_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 37), .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 37), .first, (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 19), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, .possible, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7829_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7829 row7829_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7829_block_007
  · decide
  · rfl

def row7829_block_008 : List Nat := [14621, 14723, 15017, 15473, 16193]
def plan7829_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 29), .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 11), (.firstFactor 19), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7829_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7829 row7829_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7829_block_008
  · decide
  · rfl

def row7829_block_009 : List Nat := [16763, 17489, 17957]
def plan7829_block_009 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 31), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 31), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 19), .possible, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7829_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7829 row7829_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7829_block_009
  · decide
  · rfl

def row7829_block_010 : List Nat := []
theorem row7829_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7829 row7829_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7829_block_011 : List Nat := []
theorem row7829_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7829 row7829_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7829_block_012 : List Nat := []
theorem row7829_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7829 row7829_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7829_block_013 : List Nat := []
theorem row7829_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7829 row7829_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7829_large_left_left : List Nat := row7829_small ++ row7829_block_005
theorem row7829_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7829 row7829_large_left_left :=
  Budget.RowUpperBound.branch row7829_small_covers row7829_block_005_covers

def row7829_large_left_right_right : List Nat := row7829_block_007 ++ row7829_block_008
theorem row7829_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7829 row7829_large_left_right_right :=
  Budget.RowUpperBound.branch row7829_block_007_covers row7829_block_008_covers

def row7829_large_left_right : List Nat := row7829_block_006 ++ row7829_large_left_right_right
theorem row7829_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7829 row7829_large_left_right :=
  Budget.RowUpperBound.branch row7829_block_006_covers row7829_large_left_right_right_covers

def row7829_large_left : List Nat := row7829_large_left_left ++ row7829_large_left_right
theorem row7829_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7829 row7829_large_left :=
  Budget.RowUpperBound.branch row7829_large_left_left_covers row7829_large_left_right_covers

def row7829_large_right_left : List Nat := row7829_block_009 ++ row7829_block_010
theorem row7829_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7829 row7829_large_right_left :=
  Budget.RowUpperBound.branch row7829_block_009_covers row7829_block_010_covers

def row7829_large_right_right_right : List Nat := row7829_block_012 ++ row7829_block_013
theorem row7829_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7829 row7829_large_right_right_right :=
  Budget.RowUpperBound.branch row7829_block_012_covers row7829_block_013_covers

def row7829_large_right_right : List Nat := row7829_block_011 ++ row7829_large_right_right_right
theorem row7829_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7829 row7829_large_right_right :=
  Budget.RowUpperBound.branch row7829_block_011_covers row7829_large_right_right_right_covers

def row7829_large_right : List Nat := row7829_large_right_left ++ row7829_large_right_right
theorem row7829_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7829 row7829_large_right :=
  Budget.RowUpperBound.branch row7829_large_right_left_covers row7829_large_right_right_covers

def row7829_large : List Nat := row7829_large_left ++ row7829_large_right
theorem row7829_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7829 row7829_large :=
  Budget.RowUpperBound.branch row7829_large_left_covers row7829_large_right_covers

def row7829 : List Nat := row7829_large
theorem row7829_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7829 row7829 := row7829_large_covers

end ProjectEulerSolutions.PrimePairLarge
