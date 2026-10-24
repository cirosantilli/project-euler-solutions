import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows073

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7687_small : List Nat := PrimePairUpper.smallRow7687
theorem row7687_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7687 row7687_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7687_covers

def row7687_block_005 : List Nat := [8521, 8821, 9199]
def plan7687_block_005 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .second, .first, .second, .first, .first, (.secondFactor 37), (.secondFactor 7), (.firstFactor 23), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), (.secondFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 23), (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 17), (.secondFactor 7), .first, .second, .first, (.firstFactor 37), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7687_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7687 row7687_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7687_block_005
  · decide
  · rfl

def row7687_block_006 : List Nat := [10651, 11197, 11617, 11701, 12277]
def plan7687_block_006 : List Budget.Choice := [.first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .possible, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .possible, .first, .first, (.secondFactor 23), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row7687_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7687 row7687_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7687_block_006
  · decide
  · rfl

def row7687_block_007 : List Nat := [12757, 13339, 13681, 14083, 14461]
def plan7687_block_007 : List Budget.Choice := [(.secondFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 13), (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 19), .first, .first, (.firstFactor 29), (.firstFactor 31), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 37), (.secondFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 17), .possible]
set_option maxRecDepth 2048 in
theorem row7687_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7687 row7687_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7687_block_007
  · decide
  · rfl

def row7687_block_008 : List Nat := [15823]
def plan7687_block_008 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .possible, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 29), (.secondFactor 11), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7687_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7687 row7687_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7687_block_008
  · decide
  · rfl

def row7687_block_009 : List Nat := [17359, 17581, 17977]
def plan7687_block_009 : List Budget.Choice := [(.secondFactor 17), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 19), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 13), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .second, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 23), (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .second, (.secondFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7687_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7687 row7687_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7687_block_009
  · decide
  · rfl

def row7687_block_010 : List Nat := []
theorem row7687_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7687 row7687_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7687_block_011 : List Nat := []
theorem row7687_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7687 row7687_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7687_block_012 : List Nat := []
theorem row7687_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7687 row7687_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7687_block_013 : List Nat := []
theorem row7687_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7687 row7687_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7687_large_left_left : List Nat := row7687_small ++ row7687_block_005
theorem row7687_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7687 row7687_large_left_left :=
  Budget.RowUpperBound.branch row7687_small_covers row7687_block_005_covers

def row7687_large_left_right_right : List Nat := row7687_block_007 ++ row7687_block_008
theorem row7687_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7687 row7687_large_left_right_right :=
  Budget.RowUpperBound.branch row7687_block_007_covers row7687_block_008_covers

def row7687_large_left_right : List Nat := row7687_block_006 ++ row7687_large_left_right_right
theorem row7687_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7687 row7687_large_left_right :=
  Budget.RowUpperBound.branch row7687_block_006_covers row7687_large_left_right_right_covers

def row7687_large_left : List Nat := row7687_large_left_left ++ row7687_large_left_right
theorem row7687_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7687 row7687_large_left :=
  Budget.RowUpperBound.branch row7687_large_left_left_covers row7687_large_left_right_covers

def row7687_large_right_left : List Nat := row7687_block_009 ++ row7687_block_010
theorem row7687_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7687 row7687_large_right_left :=
  Budget.RowUpperBound.branch row7687_block_009_covers row7687_block_010_covers

def row7687_large_right_right_right : List Nat := row7687_block_012 ++ row7687_block_013
theorem row7687_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7687 row7687_large_right_right_right :=
  Budget.RowUpperBound.branch row7687_block_012_covers row7687_block_013_covers

def row7687_large_right_right : List Nat := row7687_block_011 ++ row7687_large_right_right_right
theorem row7687_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7687 row7687_large_right_right :=
  Budget.RowUpperBound.branch row7687_block_011_covers row7687_large_right_right_right_covers

def row7687_large_right : List Nat := row7687_large_right_left ++ row7687_large_right_right
theorem row7687_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7687 row7687_large_right :=
  Budget.RowUpperBound.branch row7687_large_right_left_covers row7687_large_right_right_covers

def row7687_large : List Nat := row7687_large_left ++ row7687_large_right
theorem row7687_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7687 row7687_large :=
  Budget.RowUpperBound.branch row7687_large_left_covers row7687_large_right_covers

def row7687 : List Nat := row7687_large
theorem row7687_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7687 row7687 := row7687_large_covers

def row7691_small : List Nat := PrimePairUpper.smallRow7691
theorem row7691_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7691 row7691_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7691_covers

def row7691_block_005 : List Nat := [9371, 10037]
def plan7691_block_005 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .second, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7691_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7691 row7691_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7691_block_005
  · decide
  · rfl

def row7691_block_006 : List Nat := [11261, 12269]
def plan7691_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 31), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .second, .second, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 11), (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 31), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7691_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7691 row7691_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7691_block_006
  · decide
  · rfl

def row7691_block_007 : List Nat := [13049, 13757, 14369]
def plan7691_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37), .possible, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .second, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 29), .second, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 19), .first, .second, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, .second, .second, .first, (.secondFactor 31), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row7691_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7691 row7691_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7691_block_007
  · decide
  · rfl

def row7691_block_008 : List Nat := [14537, 15569, 15731]
def plan7691_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .second, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row7691_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7691 row7691_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7691_block_008
  · decide
  · rfl

def row7691_block_009 : List Nat := [16631, 16787, 17957, 18059, 18143]
def plan7691_block_009 : List Budget.Choice := [.first, (.secondFactor 13), .second, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .possible, (.secondFactor 7), .second, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 31), .first, (.secondFactor 29), (.secondFactor 23), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 11), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .second, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .second, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .possible, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), .first, .possible, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .possible, (.firstFactor 37), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .second, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7691_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7691 row7691_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7691_block_009
  · decide
  · rfl

def row7691_block_010 : List Nat := []
theorem row7691_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7691 row7691_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7691_block_011 : List Nat := []
theorem row7691_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7691 row7691_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7691_block_012 : List Nat := []
theorem row7691_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7691 row7691_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7691_block_013 : List Nat := []
theorem row7691_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7691 row7691_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7691_large_left_left : List Nat := row7691_small ++ row7691_block_005
theorem row7691_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7691 row7691_large_left_left :=
  Budget.RowUpperBound.branch row7691_small_covers row7691_block_005_covers

def row7691_large_left_right_right : List Nat := row7691_block_007 ++ row7691_block_008
theorem row7691_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7691 row7691_large_left_right_right :=
  Budget.RowUpperBound.branch row7691_block_007_covers row7691_block_008_covers

def row7691_large_left_right : List Nat := row7691_block_006 ++ row7691_large_left_right_right
theorem row7691_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7691 row7691_large_left_right :=
  Budget.RowUpperBound.branch row7691_block_006_covers row7691_large_left_right_right_covers

def row7691_large_left : List Nat := row7691_large_left_left ++ row7691_large_left_right
theorem row7691_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7691 row7691_large_left :=
  Budget.RowUpperBound.branch row7691_large_left_left_covers row7691_large_left_right_covers

def row7691_large_right_left : List Nat := row7691_block_009 ++ row7691_block_010
theorem row7691_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7691 row7691_large_right_left :=
  Budget.RowUpperBound.branch row7691_block_009_covers row7691_block_010_covers

def row7691_large_right_right_right : List Nat := row7691_block_012 ++ row7691_block_013
theorem row7691_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7691 row7691_large_right_right_right :=
  Budget.RowUpperBound.branch row7691_block_012_covers row7691_block_013_covers

def row7691_large_right_right : List Nat := row7691_block_011 ++ row7691_large_right_right_right
theorem row7691_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7691 row7691_large_right_right :=
  Budget.RowUpperBound.branch row7691_block_011_covers row7691_large_right_right_right_covers

def row7691_large_right : List Nat := row7691_large_right_left ++ row7691_large_right_right
theorem row7691_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7691 row7691_large_right :=
  Budget.RowUpperBound.branch row7691_large_right_left_covers row7691_large_right_right_covers

def row7691_large : List Nat := row7691_large_left ++ row7691_large_right
theorem row7691_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7691 row7691_large :=
  Budget.RowUpperBound.branch row7691_large_left_covers row7691_large_right_covers

def row7691 : List Nat := row7691_large
theorem row7691_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7691 row7691 := row7691_large_covers

def row7699_small : List Nat := PrimePairUpper.smallRow7699
theorem row7699_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7699 row7699_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7699_covers

def row7699_block_005 : List Nat := [9679, 9787, 10111]
def plan7699_block_005 : List Budget.Choice := [(.firstFactor 13), .second, .first, .first, (.firstFactor 19), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 31), .first, .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7699_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7699 row7699_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7699_block_005
  · decide
  · rfl

def row7699_block_006 : List Nat := [10459, 11743, 11833]
def plan7699_block_006 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 13), .first, .second, .first, .second, .first, .first, .first, (.secondFactor 11), .first, .second, (.secondFactor 17), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7699_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7699 row7699_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7699_block_006
  · decide
  · rfl

def row7699_block_007 : List Nat := [12421, 13093, 13171, 13681, 13789, 14293]
def plan7699_block_007 : List Budget.Choice := [.possible, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .second, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.firstFactor 23), .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .possible, (.firstFactor 11), (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .second, (.secondFactor 19), .possible, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .second, .second]
set_option maxRecDepth 2048 in
theorem row7699_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7699 row7699_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7699_block_007
  · decide
  · rfl

def row7699_block_008 : List Nat := [16087, 16141, 16183, 16267]
def plan7699_block_008 : List Budget.Choice := [(.firstFactor 17), .first, .second, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .second, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 13), (.secondFactor 19), .first, .first, (.secondFactor 29), (.secondFactor 23), (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 17), .first, .first, .possible, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 23), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7699_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7699 row7699_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7699_block_008
  · decide
  · rfl

def row7699_block_009 : List Nat := [16981, 17737]
def plan7699_block_009 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 37), (.secondFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 29), .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .second, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 23), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7699_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7699 row7699_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7699_block_009
  · decide
  · rfl

def row7699_block_010 : List Nat := []
theorem row7699_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7699 row7699_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7699_block_011 : List Nat := []
theorem row7699_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7699 row7699_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7699_block_012 : List Nat := []
theorem row7699_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7699 row7699_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7699_block_013 : List Nat := []
theorem row7699_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7699 row7699_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7699_large_left_left : List Nat := row7699_small ++ row7699_block_005
theorem row7699_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7699 row7699_large_left_left :=
  Budget.RowUpperBound.branch row7699_small_covers row7699_block_005_covers

def row7699_large_left_right_right : List Nat := row7699_block_007 ++ row7699_block_008
theorem row7699_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7699 row7699_large_left_right_right :=
  Budget.RowUpperBound.branch row7699_block_007_covers row7699_block_008_covers

def row7699_large_left_right : List Nat := row7699_block_006 ++ row7699_large_left_right_right
theorem row7699_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7699 row7699_large_left_right :=
  Budget.RowUpperBound.branch row7699_block_006_covers row7699_large_left_right_right_covers

def row7699_large_left : List Nat := row7699_large_left_left ++ row7699_large_left_right
theorem row7699_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7699 row7699_large_left :=
  Budget.RowUpperBound.branch row7699_large_left_left_covers row7699_large_left_right_covers

def row7699_large_right_left : List Nat := row7699_block_009 ++ row7699_block_010
theorem row7699_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7699 row7699_large_right_left :=
  Budget.RowUpperBound.branch row7699_block_009_covers row7699_block_010_covers

def row7699_large_right_right_right : List Nat := row7699_block_012 ++ row7699_block_013
theorem row7699_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7699 row7699_large_right_right_right :=
  Budget.RowUpperBound.branch row7699_block_012_covers row7699_block_013_covers

def row7699_large_right_right : List Nat := row7699_block_011 ++ row7699_large_right_right_right
theorem row7699_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7699 row7699_large_right_right :=
  Budget.RowUpperBound.branch row7699_block_011_covers row7699_large_right_right_right_covers

def row7699_large_right : List Nat := row7699_large_right_left ++ row7699_large_right_right
theorem row7699_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7699 row7699_large_right :=
  Budget.RowUpperBound.branch row7699_large_right_left_covers row7699_large_right_right_covers

def row7699_large : List Nat := row7699_large_left ++ row7699_large_right
theorem row7699_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7699 row7699_large :=
  Budget.RowUpperBound.branch row7699_large_left_covers row7699_large_right_covers

def row7699 : List Nat := row7699_large
theorem row7699_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7699 row7699 := row7699_large_covers

def row7703_small : List Nat := PrimePairUpper.smallRow7703
theorem row7703_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7703 row7703_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7703_covers

def row7703_block_005 : List Nat := []
def plan7703_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 29), .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 31), .second, .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .second, .first, .first, (.firstFactor 13), (.secondFactor 29), .first, .first, .second, (.firstFactor 37), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .second, .first, .first, (.secondFactor 19), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row7703_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7703 row7703_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7703_block_005
  · decide
  · rfl

def row7703_block_006 : List Nat := [10889, 10979, 11027, 11783, 12107, 12329]
def plan7703_block_006 : List Budget.Choice := [(.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .possible, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), .possible, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 19), (.secondFactor 17), .second, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .possible, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .second, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 23), (.secondFactor 11), (.firstFactor 17), (.firstFactor 31), .first, .first, .first, (.secondFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row7703_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7703 row7703_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7703_block_006
  · decide
  · rfl

def row7703_block_007 : List Nat := [12539, 12821, 12899, 13337, 14249, 14423]
def plan7703_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 23), .possible, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 31), (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .second, .first, .possible, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7703_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7703 row7703_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7703_block_007
  · decide
  · rfl

def row7703_block_008 : List Nat := [14717, 15263, 15731, 15761, 16103, 16139, 16361]
def plan7703_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .second, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, (.firstFactor 13), (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .second, .possible, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 13), .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 29), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row7703_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7703 row7703_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7703_block_008
  · decide
  · rfl

def row7703_block_009 : List Nat := [16553, 16823, 17519, 17579]
def plan7703_block_009 : List Budget.Choice := [.first, (.firstFactor 17), (.firstFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 11), .possible, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, .first, .first, .possible, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, .second, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 29), .second, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7703_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7703 row7703_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7703_block_009
  · decide
  · rfl

def row7703_block_010 : List Nat := []
theorem row7703_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7703 row7703_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7703_block_011 : List Nat := []
theorem row7703_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7703 row7703_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7703_block_012 : List Nat := []
theorem row7703_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7703 row7703_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7703_block_013 : List Nat := []
theorem row7703_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7703 row7703_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7703_large_left_left : List Nat := row7703_small ++ row7703_block_005
theorem row7703_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7703 row7703_large_left_left :=
  Budget.RowUpperBound.branch row7703_small_covers row7703_block_005_covers

def row7703_large_left_right_right : List Nat := row7703_block_007 ++ row7703_block_008
theorem row7703_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7703 row7703_large_left_right_right :=
  Budget.RowUpperBound.branch row7703_block_007_covers row7703_block_008_covers

def row7703_large_left_right : List Nat := row7703_block_006 ++ row7703_large_left_right_right
theorem row7703_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7703 row7703_large_left_right :=
  Budget.RowUpperBound.branch row7703_block_006_covers row7703_large_left_right_right_covers

def row7703_large_left : List Nat := row7703_large_left_left ++ row7703_large_left_right
theorem row7703_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7703 row7703_large_left :=
  Budget.RowUpperBound.branch row7703_large_left_left_covers row7703_large_left_right_covers

def row7703_large_right_left : List Nat := row7703_block_009 ++ row7703_block_010
theorem row7703_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7703 row7703_large_right_left :=
  Budget.RowUpperBound.branch row7703_block_009_covers row7703_block_010_covers

def row7703_large_right_right_right : List Nat := row7703_block_012 ++ row7703_block_013
theorem row7703_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7703 row7703_large_right_right_right :=
  Budget.RowUpperBound.branch row7703_block_012_covers row7703_block_013_covers

def row7703_large_right_right : List Nat := row7703_block_011 ++ row7703_large_right_right_right
theorem row7703_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7703 row7703_large_right_right :=
  Budget.RowUpperBound.branch row7703_block_011_covers row7703_large_right_right_right_covers

def row7703_large_right : List Nat := row7703_large_right_left ++ row7703_large_right_right
theorem row7703_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7703 row7703_large_right :=
  Budget.RowUpperBound.branch row7703_large_right_left_covers row7703_large_right_right_covers

def row7703_large : List Nat := row7703_large_left ++ row7703_large_right
theorem row7703_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7703 row7703_large :=
  Budget.RowUpperBound.branch row7703_large_left_covers row7703_large_right_covers

def row7703 : List Nat := row7703_large
theorem row7703_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7703 row7703 := row7703_large_covers

end ProjectEulerSolutions.PrimePairLarge
