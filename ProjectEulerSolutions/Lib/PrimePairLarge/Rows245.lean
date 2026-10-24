import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows074

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7753_small : List Nat := PrimePairUpper.smallRow7753
theorem row7753_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7753 row7753_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7753_covers

def row7753_block_005 : List Nat := [9109, 9283, 9661, 9871]
def plan7753_block_005 : List Budget.Choice := [.second, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .second, .possible, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .possible, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 13), .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 37), .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 29), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row7753_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7753 row7753_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7753_block_005
  · decide
  · rfl

def row7753_block_006 : List Nat := []
def plan7753_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 17), (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 19), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row7753_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7753 row7753_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7753_block_006
  · decide
  · rfl

def row7753_block_007 : List Nat := [12457, 13219, 13291, 13633, 13933, 14341]
def plan7753_block_007 : List Budget.Choice := [(.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 37), .second, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 13), .possible, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 31), (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row7753_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7753 row7753_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7753_block_007
  · decide
  · rfl

def row7753_block_008 : List Nat := [15439, 15643, 16411]
def plan7753_block_008 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .second, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 37), .possible, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 37), (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .possible, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7753_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7753 row7753_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7753_block_008
  · decide
  · rfl

def row7753_block_009 : List Nat := [17509, 18121, 18217]
def plan7753_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .possible, .first, .first, (.secondFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .possible, (.secondFactor 29), .first, (.firstFactor 17), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 17), .possible, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7753_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7753 row7753_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7753_block_009
  · decide
  · rfl

def row7753_block_010 : List Nat := []
theorem row7753_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7753 row7753_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7753_block_011 : List Nat := []
theorem row7753_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7753 row7753_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7753_block_012 : List Nat := []
theorem row7753_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7753 row7753_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7753_block_013 : List Nat := []
theorem row7753_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7753 row7753_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7753_large_left_left : List Nat := row7753_small ++ row7753_block_005
theorem row7753_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7753 row7753_large_left_left :=
  Budget.RowUpperBound.branch row7753_small_covers row7753_block_005_covers

def row7753_large_left_right_right : List Nat := row7753_block_007 ++ row7753_block_008
theorem row7753_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7753 row7753_large_left_right_right :=
  Budget.RowUpperBound.branch row7753_block_007_covers row7753_block_008_covers

def row7753_large_left_right : List Nat := row7753_block_006 ++ row7753_large_left_right_right
theorem row7753_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7753 row7753_large_left_right :=
  Budget.RowUpperBound.branch row7753_block_006_covers row7753_large_left_right_right_covers

def row7753_large_left : List Nat := row7753_large_left_left ++ row7753_large_left_right
theorem row7753_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7753 row7753_large_left :=
  Budget.RowUpperBound.branch row7753_large_left_left_covers row7753_large_left_right_covers

def row7753_large_right_left : List Nat := row7753_block_009 ++ row7753_block_010
theorem row7753_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7753 row7753_large_right_left :=
  Budget.RowUpperBound.branch row7753_block_009_covers row7753_block_010_covers

def row7753_large_right_right_right : List Nat := row7753_block_012 ++ row7753_block_013
theorem row7753_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7753 row7753_large_right_right_right :=
  Budget.RowUpperBound.branch row7753_block_012_covers row7753_block_013_covers

def row7753_large_right_right : List Nat := row7753_block_011 ++ row7753_large_right_right_right
theorem row7753_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7753 row7753_large_right_right :=
  Budget.RowUpperBound.branch row7753_block_011_covers row7753_large_right_right_right_covers

def row7753_large_right : List Nat := row7753_large_right_left ++ row7753_large_right_right
theorem row7753_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7753 row7753_large_right :=
  Budget.RowUpperBound.branch row7753_large_right_left_covers row7753_large_right_right_covers

def row7753_large : List Nat := row7753_large_left ++ row7753_large_right
theorem row7753_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7753 row7753_large :=
  Budget.RowUpperBound.branch row7753_large_left_covers row7753_large_right_covers

def row7753 : List Nat := row7753_large
theorem row7753_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7753 row7753 := row7753_large_covers

def row7757_small : List Nat := PrimePairUpper.smallRow7757
theorem row7757_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7757 row7757_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7757_covers

def row7757_block_005 : List Nat := [8597, 9209, 9419, 9521, 10193]
def plan7757_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 13), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .possible, .second, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 19), .possible, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 13), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 23), .possible, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7757_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7757 row7757_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7757_block_005
  · decide
  · rfl

def row7757_block_006 : List Nat := [11939, 12323, 12401]
def plan7757_block_006 : List Budget.Choice := [(.firstFactor 17), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 17), (.secondFactor 31), (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .second, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7757_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7757 row7757_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7757_block_006
  · decide
  · rfl

def row7757_block_007 : List Nat := []
def plan7757_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 37), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, (.secondFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .second, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 29), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 29), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7757_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7757 row7757_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7757_block_007
  · decide
  · rfl

def row7757_block_008 : List Nat := [15107, 15359, 15581, 16229]
def plan7757_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, .first, .possible, .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 29), (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 17), .second, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .possible, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row7757_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7757 row7757_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7757_block_008
  · decide
  · rfl

def row7757_block_009 : List Nat := [16547, 17231, 17627]
def plan7757_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .possible, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 23), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, .possible, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7757_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7757 row7757_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7757_block_009
  · decide
  · rfl

def row7757_block_010 : List Nat := []
theorem row7757_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7757 row7757_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7757_block_011 : List Nat := []
theorem row7757_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7757 row7757_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7757_block_012 : List Nat := []
theorem row7757_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7757 row7757_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7757_block_013 : List Nat := []
theorem row7757_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7757 row7757_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7757_large_left_left : List Nat := row7757_small ++ row7757_block_005
theorem row7757_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7757 row7757_large_left_left :=
  Budget.RowUpperBound.branch row7757_small_covers row7757_block_005_covers

def row7757_large_left_right_right : List Nat := row7757_block_007 ++ row7757_block_008
theorem row7757_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7757 row7757_large_left_right_right :=
  Budget.RowUpperBound.branch row7757_block_007_covers row7757_block_008_covers

def row7757_large_left_right : List Nat := row7757_block_006 ++ row7757_large_left_right_right
theorem row7757_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7757 row7757_large_left_right :=
  Budget.RowUpperBound.branch row7757_block_006_covers row7757_large_left_right_right_covers

def row7757_large_left : List Nat := row7757_large_left_left ++ row7757_large_left_right
theorem row7757_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7757 row7757_large_left :=
  Budget.RowUpperBound.branch row7757_large_left_left_covers row7757_large_left_right_covers

def row7757_large_right_left : List Nat := row7757_block_009 ++ row7757_block_010
theorem row7757_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7757 row7757_large_right_left :=
  Budget.RowUpperBound.branch row7757_block_009_covers row7757_block_010_covers

def row7757_large_right_right_right : List Nat := row7757_block_012 ++ row7757_block_013
theorem row7757_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7757 row7757_large_right_right_right :=
  Budget.RowUpperBound.branch row7757_block_012_covers row7757_block_013_covers

def row7757_large_right_right : List Nat := row7757_block_011 ++ row7757_large_right_right_right
theorem row7757_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7757 row7757_large_right_right :=
  Budget.RowUpperBound.branch row7757_block_011_covers row7757_large_right_right_right_covers

def row7757_large_right : List Nat := row7757_large_right_left ++ row7757_large_right_right
theorem row7757_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7757 row7757_large_right :=
  Budget.RowUpperBound.branch row7757_large_right_left_covers row7757_large_right_right_covers

def row7757_large : List Nat := row7757_large_left ++ row7757_large_right
theorem row7757_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7757 row7757_large :=
  Budget.RowUpperBound.branch row7757_large_left_covers row7757_large_right_covers

def row7757 : List Nat := row7757_large
theorem row7757_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7757 row7757 := row7757_large_covers

def row7759_small : List Nat := PrimePairUpper.smallRow7759
theorem row7759_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7759 row7759_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7759_covers

def row7759_block_005 : List Nat := [8467, 8707, 8929, 8971, 9091]
def plan7759_block_005 : List Budget.Choice := [(.firstFactor 11), (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .possible, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .possible, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .possible, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, .second, .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 37), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row7759_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7759 row7759_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7759_block_005
  · decide
  · rfl

def row7759_block_006 : List Nat := [10867, 10909, 10939, 11863, 12157, 12301]
def plan7759_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 7), .first, .possible, .first, .possible, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .second, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 37), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 19), .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row7759_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7759 row7759_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7759_block_006
  · decide
  · rfl

def row7759_block_007 : List Nat := [13381, 13687, 14143, 14431, 14437]
def plan7759_block_007 : List Budget.Choice := [(.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 29), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 31), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 13), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .possible, (.secondFactor 13), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 7), .first, .possible, .possible, .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7759_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7759 row7759_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7759_block_007
  · decide
  · rfl

def row7759_block_008 : List Nat := [15733]
def plan7759_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, .second, .first, (.secondFactor 19), .first, .second, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 17), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 37), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7759_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7759 row7759_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7759_block_008
  · decide
  · rfl

def row7759_block_009 : List Nat := [16657, 16747, 17203, 17239, 17341, 18043]
def plan7759_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 31), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 19), .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 37), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .possible, .first, .second, .first, .possible, .second, .first, (.secondFactor 23), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 11), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .second, (.secondFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7759_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7759 row7759_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7759_block_009
  · decide
  · rfl

def row7759_block_010 : List Nat := []
theorem row7759_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7759 row7759_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7759_block_011 : List Nat := []
theorem row7759_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7759 row7759_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7759_block_012 : List Nat := []
theorem row7759_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7759 row7759_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7759_block_013 : List Nat := []
theorem row7759_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7759 row7759_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7759_large_left_left : List Nat := row7759_small ++ row7759_block_005
theorem row7759_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7759 row7759_large_left_left :=
  Budget.RowUpperBound.branch row7759_small_covers row7759_block_005_covers

def row7759_large_left_right_right : List Nat := row7759_block_007 ++ row7759_block_008
theorem row7759_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7759 row7759_large_left_right_right :=
  Budget.RowUpperBound.branch row7759_block_007_covers row7759_block_008_covers

def row7759_large_left_right : List Nat := row7759_block_006 ++ row7759_large_left_right_right
theorem row7759_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7759 row7759_large_left_right :=
  Budget.RowUpperBound.branch row7759_block_006_covers row7759_large_left_right_right_covers

def row7759_large_left : List Nat := row7759_large_left_left ++ row7759_large_left_right
theorem row7759_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7759 row7759_large_left :=
  Budget.RowUpperBound.branch row7759_large_left_left_covers row7759_large_left_right_covers

def row7759_large_right_left : List Nat := row7759_block_009 ++ row7759_block_010
theorem row7759_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7759 row7759_large_right_left :=
  Budget.RowUpperBound.branch row7759_block_009_covers row7759_block_010_covers

def row7759_large_right_right_right : List Nat := row7759_block_012 ++ row7759_block_013
theorem row7759_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7759 row7759_large_right_right_right :=
  Budget.RowUpperBound.branch row7759_block_012_covers row7759_block_013_covers

def row7759_large_right_right : List Nat := row7759_block_011 ++ row7759_large_right_right_right
theorem row7759_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7759 row7759_large_right_right :=
  Budget.RowUpperBound.branch row7759_block_011_covers row7759_large_right_right_right_covers

def row7759_large_right : List Nat := row7759_large_right_left ++ row7759_large_right_right
theorem row7759_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7759 row7759_large_right :=
  Budget.RowUpperBound.branch row7759_large_right_left_covers row7759_large_right_right_covers

def row7759_large : List Nat := row7759_large_left ++ row7759_large_right
theorem row7759_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7759 row7759_large :=
  Budget.RowUpperBound.branch row7759_large_left_covers row7759_large_right_covers

def row7759 : List Nat := row7759_large
theorem row7759_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7759 row7759 := row7759_large_covers

def row7789_small : List Nat := PrimePairUpper.smallRow7789
theorem row7789_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7789 row7789_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7789_covers

def row7789_block_005 : List Nat := [8521, 8713, 9109, 9337, 9697, 9859, 9949]
def plan7789_block_005 : List Budget.Choice := [.second, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .second, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .second, .possible, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .possible, .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .second, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), .first, .possible, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row7789_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7789 row7789_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7789_block_005
  · decide
  · rfl

def row7789_block_006 : List Nat := []
def plan7789_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 37), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 29), .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row7789_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7789 row7789_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7789_block_006
  · decide
  · rfl

def row7789_block_007 : List Nat := [12763, 13249, 13477, 14281]
def plan7789_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .second, (.secondFactor 7), (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .possible, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row7789_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7789 row7789_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7789_block_007
  · decide
  · rfl

def row7789_block_008 : List Nat := [15259, 15307, 15391, 15661, 16189]
def plan7789_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 31), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 17), .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .possible, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 19), (.firstFactor 37), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, .first, .second, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 31), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7789_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7789 row7789_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7789_block_008
  · decide
  · rfl

def row7789_block_009 : List Nat := [17443, 17569]
def plan7789_block_009 : List Budget.Choice := [(.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 31), .second, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .second, .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .second, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7789_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7789 row7789_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7789_block_009
  · decide
  · rfl

def row7789_block_010 : List Nat := []
theorem row7789_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7789 row7789_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7789_block_011 : List Nat := []
theorem row7789_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7789 row7789_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7789_block_012 : List Nat := []
theorem row7789_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7789 row7789_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7789_block_013 : List Nat := []
theorem row7789_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7789 row7789_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7789_large_left_left : List Nat := row7789_small ++ row7789_block_005
theorem row7789_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7789 row7789_large_left_left :=
  Budget.RowUpperBound.branch row7789_small_covers row7789_block_005_covers

def row7789_large_left_right_right : List Nat := row7789_block_007 ++ row7789_block_008
theorem row7789_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7789 row7789_large_left_right_right :=
  Budget.RowUpperBound.branch row7789_block_007_covers row7789_block_008_covers

def row7789_large_left_right : List Nat := row7789_block_006 ++ row7789_large_left_right_right
theorem row7789_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7789 row7789_large_left_right :=
  Budget.RowUpperBound.branch row7789_block_006_covers row7789_large_left_right_right_covers

def row7789_large_left : List Nat := row7789_large_left_left ++ row7789_large_left_right
theorem row7789_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7789 row7789_large_left :=
  Budget.RowUpperBound.branch row7789_large_left_left_covers row7789_large_left_right_covers

def row7789_large_right_left : List Nat := row7789_block_009 ++ row7789_block_010
theorem row7789_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7789 row7789_large_right_left :=
  Budget.RowUpperBound.branch row7789_block_009_covers row7789_block_010_covers

def row7789_large_right_right_right : List Nat := row7789_block_012 ++ row7789_block_013
theorem row7789_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7789 row7789_large_right_right_right :=
  Budget.RowUpperBound.branch row7789_block_012_covers row7789_block_013_covers

def row7789_large_right_right : List Nat := row7789_block_011 ++ row7789_large_right_right_right
theorem row7789_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7789 row7789_large_right_right :=
  Budget.RowUpperBound.branch row7789_block_011_covers row7789_large_right_right_right_covers

def row7789_large_right : List Nat := row7789_large_right_left ++ row7789_large_right_right
theorem row7789_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7789 row7789_large_right :=
  Budget.RowUpperBound.branch row7789_large_right_left_covers row7789_large_right_right_covers

def row7789_large : List Nat := row7789_large_left ++ row7789_large_right
theorem row7789_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7789 row7789_large :=
  Budget.RowUpperBound.branch row7789_large_left_covers row7789_large_right_covers

def row7789 : List Nat := row7789_large
theorem row7789_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7789 row7789 := row7789_large_covers

end ProjectEulerSolutions.PrimePairLarge
