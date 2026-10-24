import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows074

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7717_small : List Nat := PrimePairUpper.smallRow7717
theorem row7717_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7717 row7717_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7717_covers

def row7717_block_005 : List Nat := [8389, 8467, 8719, 9103, 9901]
def plan7717_block_005 : List Budget.Choice := [.possible, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .possible, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .second, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row7717_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7717 row7717_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7717_block_005
  · decide
  · rfl

def row7717_block_006 : List Nat := [10729, 11527]
def plan7717_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 23), .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 29), .first, (.secondFactor 11), .first, .first, .first, .first, .second, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row7717_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7717 row7717_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7717_block_006
  · decide
  · rfl

def row7717_block_007 : List Nat := [12541, 12547, 13093, 13627, 14347]
def plan7717_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .possible, .possible, .first, .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 19), (.secondFactor 11), .first, .first, .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .possible, (.firstFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 37), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .possible, .first, .first, (.firstFactor 29), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7717_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7717 row7717_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7717_block_007
  · decide
  · rfl

def row7717_block_008 : List Nat := []
def plan7717_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 37), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 31), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 23), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 11), .second, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), (.firstFactor 17), (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7717_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7717 row7717_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7717_block_008
  · decide
  · rfl

def row7717_block_009 : List Nat := [17449, 18253]
def plan7717_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.secondFactor 29), (.secondFactor 17), (.firstFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, .second, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 31), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), (.secondFactor 23), (.secondFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7717_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7717 row7717_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7717_block_009
  · decide
  · rfl

def row7717_block_010 : List Nat := []
theorem row7717_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7717 row7717_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7717_block_011 : List Nat := []
theorem row7717_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7717 row7717_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7717_block_012 : List Nat := []
theorem row7717_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7717 row7717_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7717_block_013 : List Nat := []
theorem row7717_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7717 row7717_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7717_large_left_left : List Nat := row7717_small ++ row7717_block_005
theorem row7717_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7717 row7717_large_left_left :=
  Budget.RowUpperBound.branch row7717_small_covers row7717_block_005_covers

def row7717_large_left_right_right : List Nat := row7717_block_007 ++ row7717_block_008
theorem row7717_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7717 row7717_large_left_right_right :=
  Budget.RowUpperBound.branch row7717_block_007_covers row7717_block_008_covers

def row7717_large_left_right : List Nat := row7717_block_006 ++ row7717_large_left_right_right
theorem row7717_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7717 row7717_large_left_right :=
  Budget.RowUpperBound.branch row7717_block_006_covers row7717_large_left_right_right_covers

def row7717_large_left : List Nat := row7717_large_left_left ++ row7717_large_left_right
theorem row7717_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7717 row7717_large_left :=
  Budget.RowUpperBound.branch row7717_large_left_left_covers row7717_large_left_right_covers

def row7717_large_right_left : List Nat := row7717_block_009 ++ row7717_block_010
theorem row7717_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7717 row7717_large_right_left :=
  Budget.RowUpperBound.branch row7717_block_009_covers row7717_block_010_covers

def row7717_large_right_right_right : List Nat := row7717_block_012 ++ row7717_block_013
theorem row7717_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7717 row7717_large_right_right_right :=
  Budget.RowUpperBound.branch row7717_block_012_covers row7717_block_013_covers

def row7717_large_right_right : List Nat := row7717_block_011 ++ row7717_large_right_right_right
theorem row7717_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7717 row7717_large_right_right :=
  Budget.RowUpperBound.branch row7717_block_011_covers row7717_large_right_right_right_covers

def row7717_large_right : List Nat := row7717_large_right_left ++ row7717_large_right_right
theorem row7717_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7717 row7717_large_right :=
  Budget.RowUpperBound.branch row7717_large_right_left_covers row7717_large_right_right_covers

def row7717_large : List Nat := row7717_large_left ++ row7717_large_right
theorem row7717_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7717 row7717_large :=
  Budget.RowUpperBound.branch row7717_large_left_covers row7717_large_right_covers

def row7717 : List Nat := row7717_large
theorem row7717_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7717 row7717 := row7717_large_covers

def row7723_small : List Nat := PrimePairUpper.smallRow7723
theorem row7723_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7723 row7723_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7723_covers

def row7723_block_005 : List Nat := []
def plan7723_block_005 : List Budget.Choice := [(.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 37), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7723_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7723 row7723_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7723_block_005
  · decide
  · rfl

def row7723_block_006 : List Nat := [12241]
def plan7723_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .second, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row7723_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7723 row7723_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7723_block_006
  · decide
  · rfl

def row7723_block_007 : List Nat := [12577, 13033, 13291, 14461]
def plan7723_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 37), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, .possible, (.firstFactor 7), (.firstFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .second, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .possible, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .second, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .possible]
set_option maxRecDepth 2048 in
theorem row7723_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7723 row7723_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7723_block_007
  · decide
  · rfl

def row7723_block_008 : List Nat := [14533, 15601]
def plan7723_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .possible, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7723_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7723 row7723_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7723_block_008
  · decide
  · rfl

def row7723_block_009 : List Nat := [17569]
def plan7723_block_009 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 19), (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 13), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 13), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 23), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7723_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7723 row7723_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7723_block_009
  · decide
  · rfl

def row7723_block_010 : List Nat := []
theorem row7723_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7723 row7723_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7723_block_011 : List Nat := []
theorem row7723_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7723 row7723_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7723_block_012 : List Nat := []
theorem row7723_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7723 row7723_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7723_block_013 : List Nat := []
theorem row7723_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7723 row7723_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7723_large_left_left : List Nat := row7723_small ++ row7723_block_005
theorem row7723_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7723 row7723_large_left_left :=
  Budget.RowUpperBound.branch row7723_small_covers row7723_block_005_covers

def row7723_large_left_right_right : List Nat := row7723_block_007 ++ row7723_block_008
theorem row7723_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7723 row7723_large_left_right_right :=
  Budget.RowUpperBound.branch row7723_block_007_covers row7723_block_008_covers

def row7723_large_left_right : List Nat := row7723_block_006 ++ row7723_large_left_right_right
theorem row7723_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7723 row7723_large_left_right :=
  Budget.RowUpperBound.branch row7723_block_006_covers row7723_large_left_right_right_covers

def row7723_large_left : List Nat := row7723_large_left_left ++ row7723_large_left_right
theorem row7723_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7723 row7723_large_left :=
  Budget.RowUpperBound.branch row7723_large_left_left_covers row7723_large_left_right_covers

def row7723_large_right_left : List Nat := row7723_block_009 ++ row7723_block_010
theorem row7723_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7723 row7723_large_right_left :=
  Budget.RowUpperBound.branch row7723_block_009_covers row7723_block_010_covers

def row7723_large_right_right_right : List Nat := row7723_block_012 ++ row7723_block_013
theorem row7723_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7723 row7723_large_right_right_right :=
  Budget.RowUpperBound.branch row7723_block_012_covers row7723_block_013_covers

def row7723_large_right_right : List Nat := row7723_block_011 ++ row7723_large_right_right_right
theorem row7723_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7723 row7723_large_right_right :=
  Budget.RowUpperBound.branch row7723_block_011_covers row7723_large_right_right_right_covers

def row7723_large_right : List Nat := row7723_large_right_left ++ row7723_large_right_right
theorem row7723_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7723 row7723_large_right :=
  Budget.RowUpperBound.branch row7723_large_right_left_covers row7723_large_right_right_covers

def row7723_large : List Nat := row7723_large_left ++ row7723_large_right
theorem row7723_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7723 row7723_large :=
  Budget.RowUpperBound.branch row7723_large_left_covers row7723_large_right_covers

def row7723 : List Nat := row7723_large
theorem row7723_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7723 row7723 := row7723_large_covers

def row7727_small : List Nat := PrimePairUpper.smallRow7727
theorem row7727_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7727 row7727_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7727_covers

def row7727_block_005 : List Nat := [8429, 8501, 10091, 10343]
def plan7727_block_005 : List Budget.Choice := [.first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 17), .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 11), .second, (.secondFactor 17), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row7727_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7727 row7727_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7727_block_005
  · decide
  · rfl

def row7727_block_006 : List Nat := [10847, 10949]
def plan7727_block_006 : List Budget.Choice := [.second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 31), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .second, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .second, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row7727_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7727 row7727_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7727_block_006
  · decide
  · rfl

def row7727_block_007 : List Nat := [12641, 13367]
def plan7727_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 31), (.firstFactor 19), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 19), (.secondFactor 23), .second, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 37), .first, .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 11), .first, .possible, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 31), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 7), .second, .first, (.firstFactor 13), .first, (.firstFactor 37), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row7727_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7727 row7727_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7727_block_007
  · decide
  · rfl

def row7727_block_008 : List Nat := [14561, 14951, 15287]
def plan7727_block_008 : List Budget.Choice := [.first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 17), (.secondFactor 31), .first, (.firstFactor 37), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .first, .possible, .first, .second, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .second, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .second, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 29), (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .second, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 17), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row7727_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7727 row7727_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7727_block_008
  · decide
  · rfl

def row7727_block_009 : List Nat := [17987, 18269]
def plan7727_block_009 : List Budget.Choice := [.first, (.firstFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 31), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 19), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7727_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7727 row7727_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7727_block_009
  · decide
  · rfl

def row7727_block_010 : List Nat := []
theorem row7727_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7727 row7727_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7727_block_011 : List Nat := []
theorem row7727_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7727 row7727_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7727_block_012 : List Nat := []
theorem row7727_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7727 row7727_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7727_block_013 : List Nat := []
theorem row7727_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7727 row7727_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7727_large_left_left : List Nat := row7727_small ++ row7727_block_005
theorem row7727_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7727 row7727_large_left_left :=
  Budget.RowUpperBound.branch row7727_small_covers row7727_block_005_covers

def row7727_large_left_right_right : List Nat := row7727_block_007 ++ row7727_block_008
theorem row7727_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7727 row7727_large_left_right_right :=
  Budget.RowUpperBound.branch row7727_block_007_covers row7727_block_008_covers

def row7727_large_left_right : List Nat := row7727_block_006 ++ row7727_large_left_right_right
theorem row7727_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7727 row7727_large_left_right :=
  Budget.RowUpperBound.branch row7727_block_006_covers row7727_large_left_right_right_covers

def row7727_large_left : List Nat := row7727_large_left_left ++ row7727_large_left_right
theorem row7727_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7727 row7727_large_left :=
  Budget.RowUpperBound.branch row7727_large_left_left_covers row7727_large_left_right_covers

def row7727_large_right_left : List Nat := row7727_block_009 ++ row7727_block_010
theorem row7727_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7727 row7727_large_right_left :=
  Budget.RowUpperBound.branch row7727_block_009_covers row7727_block_010_covers

def row7727_large_right_right_right : List Nat := row7727_block_012 ++ row7727_block_013
theorem row7727_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7727 row7727_large_right_right_right :=
  Budget.RowUpperBound.branch row7727_block_012_covers row7727_block_013_covers

def row7727_large_right_right : List Nat := row7727_block_011 ++ row7727_large_right_right_right
theorem row7727_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7727 row7727_large_right_right :=
  Budget.RowUpperBound.branch row7727_block_011_covers row7727_large_right_right_right_covers

def row7727_large_right : List Nat := row7727_large_right_left ++ row7727_large_right_right
theorem row7727_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7727 row7727_large_right :=
  Budget.RowUpperBound.branch row7727_large_right_left_covers row7727_large_right_right_covers

def row7727_large : List Nat := row7727_large_left ++ row7727_large_right
theorem row7727_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7727 row7727_large :=
  Budget.RowUpperBound.branch row7727_large_left_covers row7727_large_right_covers

def row7727 : List Nat := row7727_large
theorem row7727_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7727 row7727 := row7727_large_covers

def row7741_small : List Nat := PrimePairUpper.smallRow7741
theorem row7741_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7741 row7741_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7741_covers

def row7741_block_005 : List Nat := [8431, 9337, 9883, 10093]
def plan7741_block_005 : List Budget.Choice := [.first, (.firstFactor 13), .first, .first, .possible, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 19), (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23), .second, .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, .second, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row7741_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7741 row7741_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7741_block_005
  · decide
  · rfl

def row7741_block_006 : List Nat := [10903]
def plan7741_block_006 : List Budget.Choice := [.first, (.secondFactor 29), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .second, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .second, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7741_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7741 row7741_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7741_block_006
  · decide
  · rfl

def row7741_block_007 : List Nat := [12589, 12601, 12613, 13147, 13597, 13831]
def plan7741_block_007 : List Budget.Choice := [(.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .possible, .possible, .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, .second, (.firstFactor 17), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 23), .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7741_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7741 row7741_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7741_block_007
  · decide
  · rfl

def row7741_block_008 : List Nat := [14683, 14713, 14983, 15061, 15583, 15679, 16033]
def plan7741_block_008 : List Budget.Choice := [(.secondFactor 11), .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, .possible, .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, (.firstFactor 37), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 29), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 31), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7741_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7741 row7741_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7741_block_008
  · decide
  · rfl

def row7741_block_009 : List Nat := [16843, 17107, 18217]
def plan7741_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 37), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 37), (.firstFactor 11), (.secondFactor 11), .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 23), (.secondFactor 13), .possible, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7741_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7741 row7741_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7741_block_009
  · decide
  · rfl

def row7741_block_010 : List Nat := []
theorem row7741_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7741 row7741_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7741_block_011 : List Nat := []
theorem row7741_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7741 row7741_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7741_block_012 : List Nat := []
theorem row7741_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7741 row7741_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7741_block_013 : List Nat := []
theorem row7741_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7741 row7741_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7741_large_left_left : List Nat := row7741_small ++ row7741_block_005
theorem row7741_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7741 row7741_large_left_left :=
  Budget.RowUpperBound.branch row7741_small_covers row7741_block_005_covers

def row7741_large_left_right_right : List Nat := row7741_block_007 ++ row7741_block_008
theorem row7741_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7741 row7741_large_left_right_right :=
  Budget.RowUpperBound.branch row7741_block_007_covers row7741_block_008_covers

def row7741_large_left_right : List Nat := row7741_block_006 ++ row7741_large_left_right_right
theorem row7741_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7741 row7741_large_left_right :=
  Budget.RowUpperBound.branch row7741_block_006_covers row7741_large_left_right_right_covers

def row7741_large_left : List Nat := row7741_large_left_left ++ row7741_large_left_right
theorem row7741_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7741 row7741_large_left :=
  Budget.RowUpperBound.branch row7741_large_left_left_covers row7741_large_left_right_covers

def row7741_large_right_left : List Nat := row7741_block_009 ++ row7741_block_010
theorem row7741_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7741 row7741_large_right_left :=
  Budget.RowUpperBound.branch row7741_block_009_covers row7741_block_010_covers

def row7741_large_right_right_right : List Nat := row7741_block_012 ++ row7741_block_013
theorem row7741_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7741 row7741_large_right_right_right :=
  Budget.RowUpperBound.branch row7741_block_012_covers row7741_block_013_covers

def row7741_large_right_right : List Nat := row7741_block_011 ++ row7741_large_right_right_right
theorem row7741_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7741 row7741_large_right_right :=
  Budget.RowUpperBound.branch row7741_block_011_covers row7741_large_right_right_right_covers

def row7741_large_right : List Nat := row7741_large_right_left ++ row7741_large_right_right
theorem row7741_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7741 row7741_large_right :=
  Budget.RowUpperBound.branch row7741_large_right_left_covers row7741_large_right_right_covers

def row7741_large : List Nat := row7741_large_left ++ row7741_large_right
theorem row7741_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7741 row7741_large :=
  Budget.RowUpperBound.branch row7741_large_left_covers row7741_large_right_covers

def row7741 : List Nat := row7741_large
theorem row7741_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7741 row7741 := row7741_large_covers

end ProjectEulerSolutions.PrimePairLarge
