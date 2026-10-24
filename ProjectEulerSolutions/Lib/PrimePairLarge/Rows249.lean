import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows076

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7907_small : List Nat := PrimePairUpper.smallRow7907
theorem row7907_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7907 row7907_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7907_covers

def row7907_block_005 : List Nat := []
def plan7907_block_005 : List Budget.Choice := [.first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 31), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .second, (.firstFactor 7), (.firstFactor 31), (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7907_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7907 row7907_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7907_block_005
  · decide
  · rfl

def row7907_block_006 : List Nat := [10889, 11027, 11681, 11807, 12107, 12119]
def plan7907_block_006 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .second, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 37), .first, .first, (.secondFactor 7), .possible, .first, (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), .first, .possible, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), .possible, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.secondFactor 19), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .possible, .first, (.firstFactor 19), .possible, (.firstFactor 13), (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .second]
set_option maxRecDepth 2048 in
theorem row7907_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7907 row7907_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7907_block_006
  · decide
  · rfl

def row7907_block_007 : List Nat := [12539, 12821, 13127, 13241]
def plan7907_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 11), .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row7907_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7907 row7907_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7907_block_007
  · decide
  · rfl

def row7907_block_008 : List Nat := [14549, 14843, 15377, 16319]
def plan7907_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 23), .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 37), (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 31), .first, (.secondFactor 29), (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 29), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .possible, .first, .first, (.firstFactor 29), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7907_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7907 row7907_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7907_block_008
  · decide
  · rfl

def row7907_block_009 : List Nat := [16607, 17117]
def plan7907_block_009 : List Budget.Choice := [.first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), .second, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 23), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .second, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7907_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7907 row7907_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7907_block_009
  · decide
  · rfl

def row7907_block_010 : List Nat := []
theorem row7907_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7907 row7907_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7907_block_011 : List Nat := []
theorem row7907_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7907 row7907_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7907_block_012 : List Nat := []
theorem row7907_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7907 row7907_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7907_block_013 : List Nat := []
theorem row7907_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7907 row7907_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7907_large_left_left : List Nat := row7907_small ++ row7907_block_005
theorem row7907_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7907 row7907_large_left_left :=
  Budget.RowUpperBound.branch row7907_small_covers row7907_block_005_covers

def row7907_large_left_right_right : List Nat := row7907_block_007 ++ row7907_block_008
theorem row7907_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7907 row7907_large_left_right_right :=
  Budget.RowUpperBound.branch row7907_block_007_covers row7907_block_008_covers

def row7907_large_left_right : List Nat := row7907_block_006 ++ row7907_large_left_right_right
theorem row7907_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7907 row7907_large_left_right :=
  Budget.RowUpperBound.branch row7907_block_006_covers row7907_large_left_right_right_covers

def row7907_large_left : List Nat := row7907_large_left_left ++ row7907_large_left_right
theorem row7907_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7907 row7907_large_left :=
  Budget.RowUpperBound.branch row7907_large_left_left_covers row7907_large_left_right_covers

def row7907_large_right_left : List Nat := row7907_block_009 ++ row7907_block_010
theorem row7907_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7907 row7907_large_right_left :=
  Budget.RowUpperBound.branch row7907_block_009_covers row7907_block_010_covers

def row7907_large_right_right_right : List Nat := row7907_block_012 ++ row7907_block_013
theorem row7907_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7907 row7907_large_right_right_right :=
  Budget.RowUpperBound.branch row7907_block_012_covers row7907_block_013_covers

def row7907_large_right_right : List Nat := row7907_block_011 ++ row7907_large_right_right_right
theorem row7907_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7907 row7907_large_right_right :=
  Budget.RowUpperBound.branch row7907_block_011_covers row7907_large_right_right_right_covers

def row7907_large_right : List Nat := row7907_large_right_left ++ row7907_large_right_right
theorem row7907_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7907 row7907_large_right :=
  Budget.RowUpperBound.branch row7907_large_right_left_covers row7907_large_right_right_covers

def row7907_large : List Nat := row7907_large_left ++ row7907_large_right
theorem row7907_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7907 row7907_large :=
  Budget.RowUpperBound.branch row7907_large_left_covers row7907_large_right_covers

def row7907 : List Nat := row7907_large
theorem row7907_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7907 row7907 := row7907_large_covers

def row7919_small : List Nat := PrimePairUpper.smallRow7919
theorem row7919_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7919 row7919_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7919_covers

def row7919_block_005 : List Nat := [9221]
def plan7919_block_005 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 17), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 19), .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7919_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7919 row7919_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7919_block_005
  · decide
  · rfl

def row7919_block_006 : List Nat := [10529, 10883, 11801, 11933, 12119]
def plan7919_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.secondFactor 31), .first, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .second, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 29), .second, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .possible, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 31)]
set_option maxRecDepth 2048 in
theorem row7919_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7919 row7919_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7919_block_006
  · decide
  · rfl

def row7919_block_007 : List Nat := [12503, 13679, 14153]
def plan7919_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 11), .possible, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 29), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .possible, .first, .first, (.secondFactor 23), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row7919_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7919 row7919_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7919_block_007
  · decide
  · rfl

def row7919_block_008 : List Nat := [16361]
def plan7919_block_008 : List Budget.Choice := [.first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 11), .first, .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 37), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 31), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7919_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7919 row7919_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7919_block_008
  · decide
  · rfl

def row7919_block_009 : List Nat := [17099]
def plan7919_block_009 : List Budget.Choice := [.first, (.secondFactor 17), .second, .second, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .second, (.secondFactor 17), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7919_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7919 row7919_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7919_block_009
  · decide
  · rfl

def row7919_block_010 : List Nat := []
theorem row7919_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7919 row7919_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7919_block_011 : List Nat := []
theorem row7919_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7919 row7919_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7919_block_012 : List Nat := []
theorem row7919_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7919 row7919_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7919_block_013 : List Nat := []
theorem row7919_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7919 row7919_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7919_large_left_left : List Nat := row7919_small ++ row7919_block_005
theorem row7919_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7919 row7919_large_left_left :=
  Budget.RowUpperBound.branch row7919_small_covers row7919_block_005_covers

def row7919_large_left_right_right : List Nat := row7919_block_007 ++ row7919_block_008
theorem row7919_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7919 row7919_large_left_right_right :=
  Budget.RowUpperBound.branch row7919_block_007_covers row7919_block_008_covers

def row7919_large_left_right : List Nat := row7919_block_006 ++ row7919_large_left_right_right
theorem row7919_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7919 row7919_large_left_right :=
  Budget.RowUpperBound.branch row7919_block_006_covers row7919_large_left_right_right_covers

def row7919_large_left : List Nat := row7919_large_left_left ++ row7919_large_left_right
theorem row7919_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7919 row7919_large_left :=
  Budget.RowUpperBound.branch row7919_large_left_left_covers row7919_large_left_right_covers

def row7919_large_right_left : List Nat := row7919_block_009 ++ row7919_block_010
theorem row7919_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7919 row7919_large_right_left :=
  Budget.RowUpperBound.branch row7919_block_009_covers row7919_block_010_covers

def row7919_large_right_right_right : List Nat := row7919_block_012 ++ row7919_block_013
theorem row7919_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7919 row7919_large_right_right_right :=
  Budget.RowUpperBound.branch row7919_block_012_covers row7919_block_013_covers

def row7919_large_right_right : List Nat := row7919_block_011 ++ row7919_large_right_right_right
theorem row7919_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7919 row7919_large_right_right :=
  Budget.RowUpperBound.branch row7919_block_011_covers row7919_large_right_right_right_covers

def row7919_large_right : List Nat := row7919_large_right_left ++ row7919_large_right_right
theorem row7919_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7919 row7919_large_right :=
  Budget.RowUpperBound.branch row7919_large_right_left_covers row7919_large_right_right_covers

def row7919_large : List Nat := row7919_large_left ++ row7919_large_right
theorem row7919_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7919 row7919_large :=
  Budget.RowUpperBound.branch row7919_large_left_covers row7919_large_right_covers

def row7919 : List Nat := row7919_large
theorem row7919_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7919 row7919 := row7919_large_covers

def row7927_small : List Nat := PrimePairUpper.smallRow7927
theorem row7927_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7927 row7927_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7927_covers

def row7927_block_005 : List Nat := [9133, 9463, 9817]
def plan7927_block_005 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 37), (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 29), .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 37), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 17), .first, .first, .second, (.firstFactor 23), (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row7927_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7927 row7927_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7927_block_005
  · decide
  · rfl

def row7927_block_006 : List Nat := [12157, 12241, 12301]
def plan7927_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .second, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37), .second, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, .first, (.firstFactor 29), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), (.secondFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .possible, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7927_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7927 row7927_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7927_block_006
  · decide
  · rfl

def row7927_block_007 : List Nat := [12721, 13309, 14107, 14347]
def plan7927_block_007 : List Budget.Choice := [.first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .second, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .second, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 23), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7927_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7927 row7927_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7927_block_007
  · decide
  · rfl

def row7927_block_008 : List Nat := [14779, 15199, 15241]
def plan7927_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, .second, .second, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .possible, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7927_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7927 row7927_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7927_block_008
  · decide
  · rfl

def row7927_block_009 : List Nat := [16657]
def plan7927_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .possible, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 29), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7927_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7927 row7927_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7927_block_009
  · decide
  · rfl

def row7927_block_010 : List Nat := []
theorem row7927_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7927 row7927_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7927_block_011 : List Nat := []
theorem row7927_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7927 row7927_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7927_block_012 : List Nat := []
theorem row7927_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7927 row7927_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7927_block_013 : List Nat := []
theorem row7927_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7927 row7927_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7927_large_left_left : List Nat := row7927_small ++ row7927_block_005
theorem row7927_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7927 row7927_large_left_left :=
  Budget.RowUpperBound.branch row7927_small_covers row7927_block_005_covers

def row7927_large_left_right_right : List Nat := row7927_block_007 ++ row7927_block_008
theorem row7927_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7927 row7927_large_left_right_right :=
  Budget.RowUpperBound.branch row7927_block_007_covers row7927_block_008_covers

def row7927_large_left_right : List Nat := row7927_block_006 ++ row7927_large_left_right_right
theorem row7927_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7927 row7927_large_left_right :=
  Budget.RowUpperBound.branch row7927_block_006_covers row7927_large_left_right_right_covers

def row7927_large_left : List Nat := row7927_large_left_left ++ row7927_large_left_right
theorem row7927_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7927 row7927_large_left :=
  Budget.RowUpperBound.branch row7927_large_left_left_covers row7927_large_left_right_covers

def row7927_large_right_left : List Nat := row7927_block_009 ++ row7927_block_010
theorem row7927_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7927 row7927_large_right_left :=
  Budget.RowUpperBound.branch row7927_block_009_covers row7927_block_010_covers

def row7927_large_right_right_right : List Nat := row7927_block_012 ++ row7927_block_013
theorem row7927_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7927 row7927_large_right_right_right :=
  Budget.RowUpperBound.branch row7927_block_012_covers row7927_block_013_covers

def row7927_large_right_right : List Nat := row7927_block_011 ++ row7927_large_right_right_right
theorem row7927_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7927 row7927_large_right_right :=
  Budget.RowUpperBound.branch row7927_block_011_covers row7927_large_right_right_right_covers

def row7927_large_right : List Nat := row7927_large_right_left ++ row7927_large_right_right
theorem row7927_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7927 row7927_large_right :=
  Budget.RowUpperBound.branch row7927_large_right_left_covers row7927_large_right_right_covers

def row7927_large : List Nat := row7927_large_left ++ row7927_large_right
theorem row7927_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7927 row7927_large :=
  Budget.RowUpperBound.branch row7927_large_left_covers row7927_large_right_covers

def row7927 : List Nat := row7927_large
theorem row7927_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7927 row7927 := row7927_large_covers

def row7933_small : List Nat := PrimePairUpper.smallRow7933
theorem row7933_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7933 row7933_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7933_covers

def row7933_block_005 : List Nat := [9133, 9613, 9643]
def plan7933_block_005 : List Budget.Choice := [(.firstFactor 13), (.secondFactor 29), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 31), .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 13), (.secondFactor 7), (.secondFactor 31), (.firstFactor 7), .possible, .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, (.secondFactor 13), .first, .first, (.firstFactor 7), .possible, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 23), (.secondFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .second]
set_option maxRecDepth 2048 in
theorem row7933_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7933 row7933_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7933_block_005
  · decide
  · rfl

def row7933_block_006 : List Nat := [10513, 10957]
def plan7933_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 31), .first, (.secondFactor 29), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.firstFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7933_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7933 row7933_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7933_block_006
  · decide
  · rfl

def row7933_block_007 : List Nat := [12517, 12601, 12697, 12781, 12853, 12889, 13903, 14281]
def plan7933_block_007 : List Budget.Choice := [(.firstFactor 11), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .second, .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 11), .first, .second, (.secondFactor 13), .first, (.firstFactor 19), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .possible, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .second, .first, .second, (.secondFactor 29), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .second, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row7933_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7933 row7933_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7933_block_007
  · decide
  · rfl

def row7933_block_008 : List Nat := []
def plan7933_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7933_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7933 row7933_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7933_block_008
  · decide
  · rfl

def row7933_block_009 : List Nat := [16603, 17443, 17977]
def plan7933_block_009 : List Budget.Choice := [(.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 31), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 37), .first, .first, .second, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 23), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 19), .possible, (.firstFactor 23), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .second, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 13), (.firstFactor 29), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .possible, .first, .first, (.secondFactor 29), (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7933_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7933 row7933_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7933_block_009
  · decide
  · rfl

def row7933_block_010 : List Nat := []
theorem row7933_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7933 row7933_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7933_block_011 : List Nat := []
theorem row7933_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7933 row7933_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7933_block_012 : List Nat := []
theorem row7933_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7933 row7933_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7933_block_013 : List Nat := []
theorem row7933_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7933 row7933_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7933_large_left_left : List Nat := row7933_small ++ row7933_block_005
theorem row7933_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7933 row7933_large_left_left :=
  Budget.RowUpperBound.branch row7933_small_covers row7933_block_005_covers

def row7933_large_left_right_right : List Nat := row7933_block_007 ++ row7933_block_008
theorem row7933_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7933 row7933_large_left_right_right :=
  Budget.RowUpperBound.branch row7933_block_007_covers row7933_block_008_covers

def row7933_large_left_right : List Nat := row7933_block_006 ++ row7933_large_left_right_right
theorem row7933_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7933 row7933_large_left_right :=
  Budget.RowUpperBound.branch row7933_block_006_covers row7933_large_left_right_right_covers

def row7933_large_left : List Nat := row7933_large_left_left ++ row7933_large_left_right
theorem row7933_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7933 row7933_large_left :=
  Budget.RowUpperBound.branch row7933_large_left_left_covers row7933_large_left_right_covers

def row7933_large_right_left : List Nat := row7933_block_009 ++ row7933_block_010
theorem row7933_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7933 row7933_large_right_left :=
  Budget.RowUpperBound.branch row7933_block_009_covers row7933_block_010_covers

def row7933_large_right_right_right : List Nat := row7933_block_012 ++ row7933_block_013
theorem row7933_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7933 row7933_large_right_right_right :=
  Budget.RowUpperBound.branch row7933_block_012_covers row7933_block_013_covers

def row7933_large_right_right : List Nat := row7933_block_011 ++ row7933_large_right_right_right
theorem row7933_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7933 row7933_large_right_right :=
  Budget.RowUpperBound.branch row7933_block_011_covers row7933_large_right_right_right_covers

def row7933_large_right : List Nat := row7933_large_right_left ++ row7933_large_right_right
theorem row7933_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7933 row7933_large_right :=
  Budget.RowUpperBound.branch row7933_large_right_left_covers row7933_large_right_right_covers

def row7933_large : List Nat := row7933_large_left ++ row7933_large_right
theorem row7933_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7933 row7933_large :=
  Budget.RowUpperBound.branch row7933_large_left_covers row7933_large_right_covers

def row7933 : List Nat := row7933_large
theorem row7933_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7933 row7933 := row7933_large_covers

end ProjectEulerSolutions.PrimePairLarge
