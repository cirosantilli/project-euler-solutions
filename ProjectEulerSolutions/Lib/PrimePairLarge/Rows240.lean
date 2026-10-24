import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows072

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7583_small : List Nat := PrimePairUpper.smallRow7583
theorem row7583_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7583 row7583_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7583_covers

def row7583_block_005 : List Nat := [8447, 9521, 9689]
def plan7583_block_005 : List Budget.Choice := [.first, .first, (.secondFactor 19), (.firstFactor 23), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 7), (.secondFactor 37), .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .second, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 31), .first, .possible, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 17), .second, (.firstFactor 31), .first, .first, .first, (.firstFactor 19), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7583_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7583 row7583_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7583_block_005
  · decide
  · rfl

def row7583_block_006 : List Nat := [10733, 11093, 11369, 11621, 11783, 11927]
def plan7583_block_006 : List Budget.Choice := [(.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), .second, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .possible, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.secondFactor 29), .first, .first, (.secondFactor 11), .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 29), .first, .possible, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .possible, .second, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row7583_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7583 row7583_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7583_block_006
  · decide
  · rfl

def row7583_block_007 : List Nat := [13487, 13613, 14321, 14327, 14387]
def plan7583_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 31), .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .possible, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .possible, .first, .possible, .first, .first, (.secondFactor 17), .possible, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row7583_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7583 row7583_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7583_block_007
  · decide
  · rfl

def row7583_block_008 : List Nat := [14561, 15401]
def plan7583_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 37), .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .second, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 31), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .possible, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .second, (.secondFactor 17), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row7583_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7583 row7583_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7583_block_008
  · decide
  · rfl

def row7583_block_009 : List Nat := [17477, 18089]
def plan7583_block_009 : List Budget.Choice := [.first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .second, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 17), .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 31), (.secondFactor 23), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 11), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .second, .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7583_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7583 row7583_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7583_block_009
  · decide
  · rfl

def row7583_block_010 : List Nat := []
theorem row7583_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7583 row7583_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7583_block_011 : List Nat := []
theorem row7583_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7583 row7583_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7583_block_012 : List Nat := []
theorem row7583_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7583 row7583_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7583_block_013 : List Nat := []
theorem row7583_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7583 row7583_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7583_large_left_left : List Nat := row7583_small ++ row7583_block_005
theorem row7583_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7583 row7583_large_left_left :=
  Budget.RowUpperBound.branch row7583_small_covers row7583_block_005_covers

def row7583_large_left_right_right : List Nat := row7583_block_007 ++ row7583_block_008
theorem row7583_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7583 row7583_large_left_right_right :=
  Budget.RowUpperBound.branch row7583_block_007_covers row7583_block_008_covers

def row7583_large_left_right : List Nat := row7583_block_006 ++ row7583_large_left_right_right
theorem row7583_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7583 row7583_large_left_right :=
  Budget.RowUpperBound.branch row7583_block_006_covers row7583_large_left_right_right_covers

def row7583_large_left : List Nat := row7583_large_left_left ++ row7583_large_left_right
theorem row7583_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7583 row7583_large_left :=
  Budget.RowUpperBound.branch row7583_large_left_left_covers row7583_large_left_right_covers

def row7583_large_right_left : List Nat := row7583_block_009 ++ row7583_block_010
theorem row7583_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7583 row7583_large_right_left :=
  Budget.RowUpperBound.branch row7583_block_009_covers row7583_block_010_covers

def row7583_large_right_right_right : List Nat := row7583_block_012 ++ row7583_block_013
theorem row7583_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7583 row7583_large_right_right_right :=
  Budget.RowUpperBound.branch row7583_block_012_covers row7583_block_013_covers

def row7583_large_right_right : List Nat := row7583_block_011 ++ row7583_large_right_right_right
theorem row7583_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7583 row7583_large_right_right :=
  Budget.RowUpperBound.branch row7583_block_011_covers row7583_large_right_right_right_covers

def row7583_large_right : List Nat := row7583_large_right_left ++ row7583_large_right_right
theorem row7583_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7583 row7583_large_right :=
  Budget.RowUpperBound.branch row7583_large_right_left_covers row7583_large_right_right_covers

def row7583_large : List Nat := row7583_large_left ++ row7583_large_right
theorem row7583_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7583 row7583_large :=
  Budget.RowUpperBound.branch row7583_large_left_covers row7583_large_right_covers

def row7583 : List Nat := row7583_large
theorem row7583_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7583 row7583 := row7583_large_covers

def row7589_small : List Nat := PrimePairUpper.smallRow7589
theorem row7589_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7589 row7589_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7589_covers

def row7589_block_005 : List Nat := [8969, 9029, 9221, 9587, 9743, 10079, 10343]
def plan7589_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 23), .possible, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 37), (.secondFactor 13), (.secondFactor 7), .first, .second, .second, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 29), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row7589_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7589 row7589_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7589_block_005
  · decide
  · rfl

def row7589_block_006 : List Nat := [10613, 11393, 11939, 12239]
def plan7589_block_006 : List Budget.Choice := [.second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .second, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 19), .first, (.secondFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .possible, (.firstFactor 17), (.firstFactor 29), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .possible, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 29), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7589_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7589 row7589_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7589_block_006
  · decide
  · rfl

def row7589_block_007 : List Nat := [13163, 13241, 14423, 14447]
def plan7589_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .second, (.firstFactor 7), .first, (.firstFactor 29), .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row7589_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7589 row7589_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7589_block_007
  · decide
  · rfl

def row7589_block_008 : List Nat := [14771, 15569, 16187]
def plan7589_block_008 : List Budget.Choice := [.first, .second, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .second, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 11), (.firstFactor 37), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .possible, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 19), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 37), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .second, (.firstFactor 7), .second]
set_option maxRecDepth 2048 in
theorem row7589_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7589 row7589_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7589_block_008
  · decide
  · rfl

def row7589_block_009 : List Nat := [17483, 17669, 18119, 18257]
def plan7589_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 19), .first, (.firstFactor 29), .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .possible, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 17), .first, .possible, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .possible, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7589_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7589 row7589_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7589_block_009
  · decide
  · rfl

def row7589_block_010 : List Nat := []
theorem row7589_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7589 row7589_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7589_block_011 : List Nat := []
theorem row7589_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7589 row7589_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7589_block_012 : List Nat := []
theorem row7589_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7589 row7589_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7589_block_013 : List Nat := []
theorem row7589_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7589 row7589_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7589_large_left_left : List Nat := row7589_small ++ row7589_block_005
theorem row7589_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7589 row7589_large_left_left :=
  Budget.RowUpperBound.branch row7589_small_covers row7589_block_005_covers

def row7589_large_left_right_right : List Nat := row7589_block_007 ++ row7589_block_008
theorem row7589_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7589 row7589_large_left_right_right :=
  Budget.RowUpperBound.branch row7589_block_007_covers row7589_block_008_covers

def row7589_large_left_right : List Nat := row7589_block_006 ++ row7589_large_left_right_right
theorem row7589_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7589 row7589_large_left_right :=
  Budget.RowUpperBound.branch row7589_block_006_covers row7589_large_left_right_right_covers

def row7589_large_left : List Nat := row7589_large_left_left ++ row7589_large_left_right
theorem row7589_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7589 row7589_large_left :=
  Budget.RowUpperBound.branch row7589_large_left_left_covers row7589_large_left_right_covers

def row7589_large_right_left : List Nat := row7589_block_009 ++ row7589_block_010
theorem row7589_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7589 row7589_large_right_left :=
  Budget.RowUpperBound.branch row7589_block_009_covers row7589_block_010_covers

def row7589_large_right_right_right : List Nat := row7589_block_012 ++ row7589_block_013
theorem row7589_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7589 row7589_large_right_right_right :=
  Budget.RowUpperBound.branch row7589_block_012_covers row7589_block_013_covers

def row7589_large_right_right : List Nat := row7589_block_011 ++ row7589_large_right_right_right
theorem row7589_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7589 row7589_large_right_right :=
  Budget.RowUpperBound.branch row7589_block_011_covers row7589_large_right_right_right_covers

def row7589_large_right : List Nat := row7589_large_right_left ++ row7589_large_right_right
theorem row7589_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7589 row7589_large_right :=
  Budget.RowUpperBound.branch row7589_large_right_left_covers row7589_large_right_right_covers

def row7589_large : List Nat := row7589_large_left ++ row7589_large_right
theorem row7589_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7589 row7589_large :=
  Budget.RowUpperBound.branch row7589_large_left_covers row7589_large_right_covers

def row7589 : List Nat := row7589_large
theorem row7589_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7589 row7589 := row7589_large_covers

def row7591_small : List Nat := PrimePairUpper.smallRow7591
theorem row7591_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7591 row7591_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7591_covers

def row7591_block_005 : List Nat := [8431, 8581, 9049, 9631, 10159]
def plan7591_block_005 : List Budget.Choice := [.second, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 29), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 37), (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 29), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row7591_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7591 row7591_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7591_block_005
  · decide
  · rfl

def row7591_block_006 : List Nat := [10909, 11827, 12241]
def plan7591_block_006 : List Budget.Choice := [.first, (.secondFactor 31), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 23), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, (.secondFactor 17), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 13), .possible, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .second, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .second, (.firstFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 19), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row7591_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7591 row7591_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7591_block_006
  · decide
  · rfl

def row7591_block_007 : List Nat := [14347]
def plan7591_block_007 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.secondFactor 11), (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 37), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 17), .second, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 31), (.firstFactor 23), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, .possible, .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7591_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7591 row7591_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7591_block_007
  · decide
  · rfl

def row7591_block_008 : List Nat := [15733]
def plan7591_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 13), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .possible, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), (.secondFactor 11), .second, .first, .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7591_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7591 row7591_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7591_block_008
  · decide
  · rfl

def row7591_block_009 : List Nat := [16903, 17383, 17449, 17761, 17791]
def plan7591_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 19), (.firstFactor 7), .possible, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 23), .second, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7591_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7591 row7591_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7591_block_009
  · decide
  · rfl

def row7591_block_010 : List Nat := []
theorem row7591_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7591 row7591_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7591_block_011 : List Nat := []
theorem row7591_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7591 row7591_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7591_block_012 : List Nat := []
theorem row7591_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7591 row7591_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7591_block_013 : List Nat := []
theorem row7591_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7591 row7591_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7591_large_left_left : List Nat := row7591_small ++ row7591_block_005
theorem row7591_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7591 row7591_large_left_left :=
  Budget.RowUpperBound.branch row7591_small_covers row7591_block_005_covers

def row7591_large_left_right_right : List Nat := row7591_block_007 ++ row7591_block_008
theorem row7591_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7591 row7591_large_left_right_right :=
  Budget.RowUpperBound.branch row7591_block_007_covers row7591_block_008_covers

def row7591_large_left_right : List Nat := row7591_block_006 ++ row7591_large_left_right_right
theorem row7591_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7591 row7591_large_left_right :=
  Budget.RowUpperBound.branch row7591_block_006_covers row7591_large_left_right_right_covers

def row7591_large_left : List Nat := row7591_large_left_left ++ row7591_large_left_right
theorem row7591_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7591 row7591_large_left :=
  Budget.RowUpperBound.branch row7591_large_left_left_covers row7591_large_left_right_covers

def row7591_large_right_left : List Nat := row7591_block_009 ++ row7591_block_010
theorem row7591_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7591 row7591_large_right_left :=
  Budget.RowUpperBound.branch row7591_block_009_covers row7591_block_010_covers

def row7591_large_right_right_right : List Nat := row7591_block_012 ++ row7591_block_013
theorem row7591_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7591 row7591_large_right_right_right :=
  Budget.RowUpperBound.branch row7591_block_012_covers row7591_block_013_covers

def row7591_large_right_right : List Nat := row7591_block_011 ++ row7591_large_right_right_right
theorem row7591_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7591 row7591_large_right_right :=
  Budget.RowUpperBound.branch row7591_block_011_covers row7591_large_right_right_right_covers

def row7591_large_right : List Nat := row7591_large_right_left ++ row7591_large_right_right
theorem row7591_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7591 row7591_large_right :=
  Budget.RowUpperBound.branch row7591_large_right_left_covers row7591_large_right_right_covers

def row7591_large : List Nat := row7591_large_left ++ row7591_large_right
theorem row7591_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7591 row7591_large :=
  Budget.RowUpperBound.branch row7591_large_left_covers row7591_large_right_covers

def row7591 : List Nat := row7591_large
theorem row7591_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7591 row7591 := row7591_large_covers

def row7603_small : List Nat := PrimePairUpper.smallRow7603
theorem row7603_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7603 row7603_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7603_covers

def row7603_block_005 : List Nat := [8731, 8971, 9631]
def plan7603_block_005 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 13), (.secondFactor 29), .first, .second, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 31), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, .possible, (.firstFactor 37), .first, .first, .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .second, (.firstFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .second, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .possible, .second, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 23), .first, .first, (.secondFactor 31), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7603_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7603 row7603_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7603_block_005
  · decide
  · rfl

def row7603_block_006 : List Nat := [10501, 11173, 11971]
def plan7603_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.secondFactor 11), .second, (.firstFactor 13), .first, (.firstFactor 23), .first, .possible, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 29), (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7603_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7603 row7603_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7603_block_006
  · decide
  · rfl

def row7603_block_007 : List Nat := [12697, 12841, 13597, 14323]
def plan7603_block_007 : List Budget.Choice := [(.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .possible, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .second, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 23), .possible, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .second, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row7603_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7603 row7603_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7603_block_007
  · decide
  · rfl

def row7603_block_008 : List Nat := [14767, 14827, 14869, 15289, 15679, 15733]
def plan7603_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .possible, .first, (.secondFactor 7), (.secondFactor 29), (.secondFactor 13), .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 29), .first, (.secondFactor 17), .first, .first, (.firstFactor 31), .second, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 19), (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7603_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7603 row7603_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7603_block_008
  · decide
  · rfl

def row7603_block_009 : List Nat := [16741, 17863]
def plan7603_block_009 : List Budget.Choice := [(.firstFactor 29), .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .second, .first, (.secondFactor 11), .possible, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 19), .possible, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7603_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7603 row7603_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7603_block_009
  · decide
  · rfl

def row7603_block_010 : List Nat := []
theorem row7603_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7603 row7603_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7603_block_011 : List Nat := []
theorem row7603_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7603 row7603_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7603_block_012 : List Nat := []
theorem row7603_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7603 row7603_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7603_block_013 : List Nat := []
theorem row7603_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7603 row7603_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7603_large_left_left : List Nat := row7603_small ++ row7603_block_005
theorem row7603_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7603 row7603_large_left_left :=
  Budget.RowUpperBound.branch row7603_small_covers row7603_block_005_covers

def row7603_large_left_right_right : List Nat := row7603_block_007 ++ row7603_block_008
theorem row7603_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7603 row7603_large_left_right_right :=
  Budget.RowUpperBound.branch row7603_block_007_covers row7603_block_008_covers

def row7603_large_left_right : List Nat := row7603_block_006 ++ row7603_large_left_right_right
theorem row7603_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7603 row7603_large_left_right :=
  Budget.RowUpperBound.branch row7603_block_006_covers row7603_large_left_right_right_covers

def row7603_large_left : List Nat := row7603_large_left_left ++ row7603_large_left_right
theorem row7603_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7603 row7603_large_left :=
  Budget.RowUpperBound.branch row7603_large_left_left_covers row7603_large_left_right_covers

def row7603_large_right_left : List Nat := row7603_block_009 ++ row7603_block_010
theorem row7603_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7603 row7603_large_right_left :=
  Budget.RowUpperBound.branch row7603_block_009_covers row7603_block_010_covers

def row7603_large_right_right_right : List Nat := row7603_block_012 ++ row7603_block_013
theorem row7603_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7603 row7603_large_right_right_right :=
  Budget.RowUpperBound.branch row7603_block_012_covers row7603_block_013_covers

def row7603_large_right_right : List Nat := row7603_block_011 ++ row7603_large_right_right_right
theorem row7603_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7603 row7603_large_right_right :=
  Budget.RowUpperBound.branch row7603_block_011_covers row7603_large_right_right_right_covers

def row7603_large_right : List Nat := row7603_large_right_left ++ row7603_large_right_right
theorem row7603_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7603 row7603_large_right :=
  Budget.RowUpperBound.branch row7603_large_right_left_covers row7603_large_right_right_covers

def row7603_large : List Nat := row7603_large_left ++ row7603_large_right
theorem row7603_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7603 row7603_large :=
  Budget.RowUpperBound.branch row7603_large_left_covers row7603_large_right_covers

def row7603 : List Nat := row7603_large
theorem row7603_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7603 row7603 := row7603_large_covers

end ProjectEulerSolutions.PrimePairLarge
