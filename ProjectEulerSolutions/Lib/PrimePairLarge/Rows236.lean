import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows070

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7481_small : List Nat := PrimePairUpper.smallRow7481
theorem row7481_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7481 row7481_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7481_covers

def row7481_block_005 : List Nat := [8741, 9413, 9551, 9887, 10037]
def plan7481_block_005 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .possible, (.firstFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, (.firstFactor 19), (.firstFactor 7), (.secondFactor 23), .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .possible, (.firstFactor 19), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 37), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 11), (.secondFactor 19), .first, (.firstFactor 7), .second, (.firstFactor 17), (.firstFactor 7), .first, .second, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7481_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7481 row7481_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7481_block_005
  · decide
  · rfl

def row7481_block_006 : List Nat := [10499, 10607, 11369, 11597, 11783, 12041]
def plan7481_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 29), .first, .first, .first, (.firstFactor 37), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), (.secondFactor 19), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .possible, .first, (.firstFactor 37), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 23), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 13), .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 29), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .second, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row7481_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7481 row7481_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7481_block_006
  · decide
  · rfl

def row7481_block_007 : List Nat := [12917, 13127, 13229, 13649, 13763]
def plan7481_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .possible, .first, (.secondFactor 31), (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 19), .first, .first, .first, .second, (.secondFactor 31), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .possible, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .second, (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, .second, .first, .first, (.secondFactor 17), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7481_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7481 row7481_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7481_block_007
  · decide
  · rfl

def row7481_block_008 : List Nat := [14621, 15959]
def plan7481_block_008 : List Budget.Choice := [.first, .second, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 29), .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 11), .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 29), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7481_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7481 row7481_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7481_block_008
  · decide
  · rfl

def row7481_block_009 : List Nat := [16619, 17417, 17477, 17729, 18311]
def plan7481_block_009 : List Budget.Choice := [.first, (.secondFactor 23), (.secondFactor 13), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .second, .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 37), .first, .possible, .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.secondFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .second, .second, .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 13), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .outside]
set_option maxRecDepth 2048 in
theorem row7481_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7481 row7481_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7481_block_009
  · decide
  · rfl

def row7481_block_010 : List Nat := []
theorem row7481_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7481 row7481_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7481_block_011 : List Nat := []
theorem row7481_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7481 row7481_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7481_block_012 : List Nat := []
theorem row7481_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7481 row7481_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7481_block_013 : List Nat := []
theorem row7481_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7481 row7481_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7481_large_left_left : List Nat := row7481_small ++ row7481_block_005
theorem row7481_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7481 row7481_large_left_left :=
  Budget.RowUpperBound.branch row7481_small_covers row7481_block_005_covers

def row7481_large_left_right_right : List Nat := row7481_block_007 ++ row7481_block_008
theorem row7481_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7481 row7481_large_left_right_right :=
  Budget.RowUpperBound.branch row7481_block_007_covers row7481_block_008_covers

def row7481_large_left_right : List Nat := row7481_block_006 ++ row7481_large_left_right_right
theorem row7481_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7481 row7481_large_left_right :=
  Budget.RowUpperBound.branch row7481_block_006_covers row7481_large_left_right_right_covers

def row7481_large_left : List Nat := row7481_large_left_left ++ row7481_large_left_right
theorem row7481_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7481 row7481_large_left :=
  Budget.RowUpperBound.branch row7481_large_left_left_covers row7481_large_left_right_covers

def row7481_large_right_left : List Nat := row7481_block_009 ++ row7481_block_010
theorem row7481_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7481 row7481_large_right_left :=
  Budget.RowUpperBound.branch row7481_block_009_covers row7481_block_010_covers

def row7481_large_right_right_right : List Nat := row7481_block_012 ++ row7481_block_013
theorem row7481_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7481 row7481_large_right_right_right :=
  Budget.RowUpperBound.branch row7481_block_012_covers row7481_block_013_covers

def row7481_large_right_right : List Nat := row7481_block_011 ++ row7481_large_right_right_right
theorem row7481_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7481 row7481_large_right_right :=
  Budget.RowUpperBound.branch row7481_block_011_covers row7481_large_right_right_right_covers

def row7481_large_right : List Nat := row7481_large_right_left ++ row7481_large_right_right
theorem row7481_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7481 row7481_large_right :=
  Budget.RowUpperBound.branch row7481_large_right_left_covers row7481_large_right_right_covers

def row7481_large : List Nat := row7481_large_left ++ row7481_large_right
theorem row7481_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7481 row7481_large :=
  Budget.RowUpperBound.branch row7481_large_left_covers row7481_large_right_covers

def row7481 : List Nat := row7481_large
theorem row7481_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7481 row7481 := row7481_large_covers

def row7487_small : List Nat := PrimePairUpper.smallRow7487
theorem row7487_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7487 row7487_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7487_covers

def row7487_block_005 : List Nat := [9011, 9923]
def plan7487_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .second, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .second, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 13), .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 17), (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .second, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7487_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7487 row7487_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7487_block_005
  · decide
  · rfl

def row7487_block_006 : List Nat := [10433, 10463, 11261, 11273, 11351, 12101]
def plan7487_block_006 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .second, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 29), (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .possible, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 7), .first, .second, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 23), (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .second, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 23), (.secondFactor 11), .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .second, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row7487_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7487 row7487_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7487_block_006
  · decide
  · rfl

def row7487_block_007 : List Nat := [13709, 14303]
def plan7487_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 19), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .second, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .possible, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row7487_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7487 row7487_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7487_block_007
  · decide
  · rfl

def row7487_block_008 : List Nat := [14723, 15101, 15467, 15641, 15797]
def plan7487_block_008 : List Budget.Choice := [.first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 23), .first, .first, .possible, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .possible, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 7), .possible, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 19), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7487_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7487 row7487_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7487_block_008
  · decide
  · rfl

def row7487_block_009 : List Nat := [16979, 17159, 18041]
def plan7487_block_009 : List Budget.Choice := [.first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), .first, .possible, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .outside]
set_option maxRecDepth 2048 in
theorem row7487_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7487 row7487_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7487_block_009
  · decide
  · rfl

def row7487_block_010 : List Nat := []
theorem row7487_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7487 row7487_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7487_block_011 : List Nat := []
theorem row7487_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7487 row7487_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7487_block_012 : List Nat := []
theorem row7487_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7487 row7487_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7487_block_013 : List Nat := []
theorem row7487_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7487 row7487_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7487_large_left_left : List Nat := row7487_small ++ row7487_block_005
theorem row7487_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7487 row7487_large_left_left :=
  Budget.RowUpperBound.branch row7487_small_covers row7487_block_005_covers

def row7487_large_left_right_right : List Nat := row7487_block_007 ++ row7487_block_008
theorem row7487_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7487 row7487_large_left_right_right :=
  Budget.RowUpperBound.branch row7487_block_007_covers row7487_block_008_covers

def row7487_large_left_right : List Nat := row7487_block_006 ++ row7487_large_left_right_right
theorem row7487_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7487 row7487_large_left_right :=
  Budget.RowUpperBound.branch row7487_block_006_covers row7487_large_left_right_right_covers

def row7487_large_left : List Nat := row7487_large_left_left ++ row7487_large_left_right
theorem row7487_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7487 row7487_large_left :=
  Budget.RowUpperBound.branch row7487_large_left_left_covers row7487_large_left_right_covers

def row7487_large_right_left : List Nat := row7487_block_009 ++ row7487_block_010
theorem row7487_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7487 row7487_large_right_left :=
  Budget.RowUpperBound.branch row7487_block_009_covers row7487_block_010_covers

def row7487_large_right_right_right : List Nat := row7487_block_012 ++ row7487_block_013
theorem row7487_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7487 row7487_large_right_right_right :=
  Budget.RowUpperBound.branch row7487_block_012_covers row7487_block_013_covers

def row7487_large_right_right : List Nat := row7487_block_011 ++ row7487_large_right_right_right
theorem row7487_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7487 row7487_large_right_right :=
  Budget.RowUpperBound.branch row7487_block_011_covers row7487_large_right_right_right_covers

def row7487_large_right : List Nat := row7487_large_right_left ++ row7487_large_right_right
theorem row7487_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7487 row7487_large_right :=
  Budget.RowUpperBound.branch row7487_large_right_left_covers row7487_large_right_right_covers

def row7487_large : List Nat := row7487_large_left ++ row7487_large_right
theorem row7487_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7487 row7487_large :=
  Budget.RowUpperBound.branch row7487_large_left_covers row7487_large_right_covers

def row7487 : List Nat := row7487_large
theorem row7487_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7487 row7487 := row7487_large_covers

def row7489_small : List Nat := PrimePairUpper.smallRow7489
theorem row7489_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7489 row7489_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7489_covers

def row7489_block_005 : List Nat := [8443, 8581, 8737, 8839, 9043, 9049, 9157, 9343, 9631, 10039, 10093]
def plan7489_block_005 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 31), .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .possible, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 29), (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .possible, .possible, .first, (.secondFactor 7), (.secondFactor 13), .second, (.firstFactor 23), (.firstFactor 17), .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 31), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 37), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row7489_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7489 row7489_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7489_block_005
  · decide
  · rfl

def row7489_block_006 : List Nat := [10531, 11287, 11887, 12253]
def plan7489_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 23), .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .possible, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .second, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), .second, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), (.secondFactor 17), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .second, .first, (.secondFactor 11), (.firstFactor 37), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .second, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7489_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7489 row7489_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7489_block_006
  · decide
  · rfl

def row7489_block_007 : List Nat := [13171, 14029, 14143]
def plan7489_block_007 : List Budget.Choice := [(.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 19), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .possible, (.firstFactor 19), (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), .possible, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 29), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 19), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row7489_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7489 row7489_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7489_block_007
  · decide
  · rfl

def row7489_block_008 : List Nat := [14827, 15439]
def plan7489_block_008 : List Budget.Choice := [(.secondFactor 23), .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), .possible, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 13), (.secondFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 19), .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 11), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7489_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7489 row7489_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7489_block_008
  · decide
  · rfl

def row7489_block_009 : List Nat := [16699, 17419, 17539, 17977, 17989, 18043]
def plan7489_block_009 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 37), (.secondFactor 17), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .possible, (.secondFactor 7), (.secondFactor 11), .first, .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .possible, (.secondFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 19), (.secondFactor 13), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 11), (.firstFactor 23), (.secondFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .outside]
set_option maxRecDepth 2048 in
theorem row7489_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7489 row7489_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7489_block_009
  · decide
  · rfl

def row7489_block_010 : List Nat := []
theorem row7489_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7489 row7489_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7489_block_011 : List Nat := []
theorem row7489_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7489 row7489_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7489_block_012 : List Nat := []
theorem row7489_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7489 row7489_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7489_block_013 : List Nat := []
theorem row7489_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7489 row7489_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7489_large_left_left : List Nat := row7489_small ++ row7489_block_005
theorem row7489_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7489 row7489_large_left_left :=
  Budget.RowUpperBound.branch row7489_small_covers row7489_block_005_covers

def row7489_large_left_right_right : List Nat := row7489_block_007 ++ row7489_block_008
theorem row7489_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7489 row7489_large_left_right_right :=
  Budget.RowUpperBound.branch row7489_block_007_covers row7489_block_008_covers

def row7489_large_left_right : List Nat := row7489_block_006 ++ row7489_large_left_right_right
theorem row7489_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7489 row7489_large_left_right :=
  Budget.RowUpperBound.branch row7489_block_006_covers row7489_large_left_right_right_covers

def row7489_large_left : List Nat := row7489_large_left_left ++ row7489_large_left_right
theorem row7489_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7489 row7489_large_left :=
  Budget.RowUpperBound.branch row7489_large_left_left_covers row7489_large_left_right_covers

def row7489_large_right_left : List Nat := row7489_block_009 ++ row7489_block_010
theorem row7489_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7489 row7489_large_right_left :=
  Budget.RowUpperBound.branch row7489_block_009_covers row7489_block_010_covers

def row7489_large_right_right_right : List Nat := row7489_block_012 ++ row7489_block_013
theorem row7489_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7489 row7489_large_right_right_right :=
  Budget.RowUpperBound.branch row7489_block_012_covers row7489_block_013_covers

def row7489_large_right_right : List Nat := row7489_block_011 ++ row7489_large_right_right_right
theorem row7489_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7489 row7489_large_right_right :=
  Budget.RowUpperBound.branch row7489_block_011_covers row7489_large_right_right_right_covers

def row7489_large_right : List Nat := row7489_large_right_left ++ row7489_large_right_right
theorem row7489_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7489 row7489_large_right :=
  Budget.RowUpperBound.branch row7489_large_right_left_covers row7489_large_right_right_covers

def row7489_large : List Nat := row7489_large_left ++ row7489_large_right
theorem row7489_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7489 row7489_large :=
  Budget.RowUpperBound.branch row7489_large_left_covers row7489_large_right_covers

def row7489 : List Nat := row7489_large
theorem row7489_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7489 row7489 := row7489_large_covers

def row7499_small : List Nat := PrimePairUpper.smallRow7499
theorem row7499_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7499 row7499_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7499_covers

def row7499_block_005 : List Nat := [9059, 9857]
def plan7499_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 17), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 37), .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 17), .first, (.secondFactor 7), .second, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 31), (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .second, (.firstFactor 37), .possible, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 31), (.firstFactor 23), (.secondFactor 11), (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7499_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7499 row7499_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7499_block_005
  · decide
  · rfl

def row7499_block_006 : List Nat := [11447, 11717, 11831, 11867, 12377]
def plan7499_block_006 : List Budget.Choice := [(.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, .second, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .possible, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row7499_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7499 row7499_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7499_block_006
  · decide
  · rfl

def row7499_block_007 : List Nat := [13763, 13859, 14153]
def plan7499_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .second, (.firstFactor 29), .first, (.secondFactor 7), .second, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 29), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 7), .second, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .possible, (.firstFactor 23), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 11), .possible, .first, (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 29), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .possible, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7499_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7499 row7499_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7499_block_007
  · decide
  · rfl

def row7499_block_008 : List Nat := [14771, 15077, 15629]
def plan7499_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .possible, .first, (.secondFactor 19), .first, (.secondFactor 31), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 29), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 29), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row7499_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7499 row7499_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7499_block_008
  · decide
  · rfl

def row7499_block_009 : List Nat := [16763, 17939, 18149, 18443]
def plan7499_block_009 : List Budget.Choice := [.first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 31), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .possible, (.secondFactor 17), (.firstFactor 7), .second, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .second, .first, .first, .second, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .possible, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .possible, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7499_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7499 row7499_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7499_block_009
  · decide
  · rfl

def row7499_block_010 : List Nat := []
theorem row7499_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7499 row7499_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7499_block_011 : List Nat := []
theorem row7499_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7499 row7499_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7499_block_012 : List Nat := []
theorem row7499_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7499 row7499_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7499_block_013 : List Nat := []
theorem row7499_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7499 row7499_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7499_large_left_left : List Nat := row7499_small ++ row7499_block_005
theorem row7499_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7499 row7499_large_left_left :=
  Budget.RowUpperBound.branch row7499_small_covers row7499_block_005_covers

def row7499_large_left_right_right : List Nat := row7499_block_007 ++ row7499_block_008
theorem row7499_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7499 row7499_large_left_right_right :=
  Budget.RowUpperBound.branch row7499_block_007_covers row7499_block_008_covers

def row7499_large_left_right : List Nat := row7499_block_006 ++ row7499_large_left_right_right
theorem row7499_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7499 row7499_large_left_right :=
  Budget.RowUpperBound.branch row7499_block_006_covers row7499_large_left_right_right_covers

def row7499_large_left : List Nat := row7499_large_left_left ++ row7499_large_left_right
theorem row7499_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7499 row7499_large_left :=
  Budget.RowUpperBound.branch row7499_large_left_left_covers row7499_large_left_right_covers

def row7499_large_right_left : List Nat := row7499_block_009 ++ row7499_block_010
theorem row7499_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7499 row7499_large_right_left :=
  Budget.RowUpperBound.branch row7499_block_009_covers row7499_block_010_covers

def row7499_large_right_right_right : List Nat := row7499_block_012 ++ row7499_block_013
theorem row7499_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7499 row7499_large_right_right_right :=
  Budget.RowUpperBound.branch row7499_block_012_covers row7499_block_013_covers

def row7499_large_right_right : List Nat := row7499_block_011 ++ row7499_large_right_right_right
theorem row7499_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7499 row7499_large_right_right :=
  Budget.RowUpperBound.branch row7499_block_011_covers row7499_large_right_right_right_covers

def row7499_large_right : List Nat := row7499_large_right_left ++ row7499_large_right_right
theorem row7499_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7499 row7499_large_right :=
  Budget.RowUpperBound.branch row7499_large_right_left_covers row7499_large_right_right_covers

def row7499_large : List Nat := row7499_large_left ++ row7499_large_right
theorem row7499_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7499 row7499_large :=
  Budget.RowUpperBound.branch row7499_large_left_covers row7499_large_right_covers

def row7499 : List Nat := row7499_large
theorem row7499_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7499 row7499 := row7499_large_covers

end ProjectEulerSolutions.PrimePairLarge
