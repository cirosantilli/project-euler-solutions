import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9221_small : List Nat := []
theorem row9221_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9221 row9221_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9221_block_005 : List Nat := [9377, 9791, 9851, 10067]
def plan9221_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 37), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .possible, .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 29), .first, .first, .first, .possible, (.firstFactor 13), .first, .first, .first, .second, (.secondFactor 23), .possible, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 13), (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 19), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9221_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9221 row9221_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9221_block_005
  · decide
  · rfl

def row9221_block_006 : List Nat := [10613]
def plan9221_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 29), .first, (.secondFactor 19), .second, .first, .first, (.secondFactor 23), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.secondFactor 11), (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 29), .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 31)]
set_option maxRecDepth 2048 in
theorem row9221_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9221 row9221_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9221_block_006
  · decide
  · rfl

def row9221_block_007 : List Nat := [12479, 12809, 12923, 13103, 13691, 14177]
def plan9221_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .second, .second, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .possible, (.firstFactor 23), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 11), (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 23), .first, .second, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row9221_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9221 row9221_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9221_block_007
  · decide
  · rfl

def row9221_block_008 : List Nat := [15737]
def plan9221_block_008 : List Budget.Choice := [.first, .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .possible, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 19), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row9221_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9221 row9221_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9221_block_008
  · decide
  · rfl

def row9221_block_009 : List Nat := [16787]
def plan9221_block_009 : List Budget.Choice := [.first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9221_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9221 row9221_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9221_block_009
  · decide
  · rfl

def row9221_block_010 : List Nat := []
theorem row9221_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9221 row9221_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9221_block_011 : List Nat := []
theorem row9221_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9221 row9221_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9221_block_012 : List Nat := []
theorem row9221_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9221 row9221_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9221_block_013 : List Nat := []
theorem row9221_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9221 row9221_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9221_large_left_left : List Nat := row9221_small ++ row9221_block_005
theorem row9221_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9221 row9221_large_left_left :=
  Budget.RowUpperBound.branch row9221_small_covers row9221_block_005_covers

def row9221_large_left_right_right : List Nat := row9221_block_007 ++ row9221_block_008
theorem row9221_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9221 row9221_large_left_right_right :=
  Budget.RowUpperBound.branch row9221_block_007_covers row9221_block_008_covers

def row9221_large_left_right : List Nat := row9221_block_006 ++ row9221_large_left_right_right
theorem row9221_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9221 row9221_large_left_right :=
  Budget.RowUpperBound.branch row9221_block_006_covers row9221_large_left_right_right_covers

def row9221_large_left : List Nat := row9221_large_left_left ++ row9221_large_left_right
theorem row9221_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9221 row9221_large_left :=
  Budget.RowUpperBound.branch row9221_large_left_left_covers row9221_large_left_right_covers

def row9221_large_right_left : List Nat := row9221_block_009 ++ row9221_block_010
theorem row9221_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9221 row9221_large_right_left :=
  Budget.RowUpperBound.branch row9221_block_009_covers row9221_block_010_covers

def row9221_large_right_right_right : List Nat := row9221_block_012 ++ row9221_block_013
theorem row9221_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9221 row9221_large_right_right_right :=
  Budget.RowUpperBound.branch row9221_block_012_covers row9221_block_013_covers

def row9221_large_right_right : List Nat := row9221_block_011 ++ row9221_large_right_right_right
theorem row9221_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9221 row9221_large_right_right :=
  Budget.RowUpperBound.branch row9221_block_011_covers row9221_large_right_right_right_covers

def row9221_large_right : List Nat := row9221_large_right_left ++ row9221_large_right_right
theorem row9221_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9221 row9221_large_right :=
  Budget.RowUpperBound.branch row9221_large_right_left_covers row9221_large_right_right_covers

def row9221_large : List Nat := row9221_large_left ++ row9221_large_right
theorem row9221_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9221 row9221_large :=
  Budget.RowUpperBound.branch row9221_large_left_covers row9221_large_right_covers

def row9221 : List Nat := row9221_large
theorem row9221_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9221 row9221 := row9221_large_covers

def row9227_small : List Nat := []
theorem row9227_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9227 row9227_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9227_block_005 : List Nat := [9311, 9461, 10247]
def plan9227_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 37), .possible, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .second, .first, .second, .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 37), (.secondFactor 13), (.secondFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row9227_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9227 row9227_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9227_block_005
  · decide
  · rfl

def row9227_block_006 : List Nat := [10457, 11087, 11351, 11423, 11483, 11549, 12149]
def plan9227_block_006 : List Budget.Choice := [(.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 29), (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 29), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .possible, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 19), .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 29), .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9227_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9227 row9227_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9227_block_006
  · decide
  · rfl

def row9227_block_007 : List Nat := [12491]
def plan9227_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 31), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 31), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 29), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row9227_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9227 row9227_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9227_block_007
  · decide
  · rfl

def row9227_block_008 : List Nat := [14657, 15329, 15569, 15767, 16187, 16451]
def plan9227_block_008 : List Budget.Choice := [.first, (.firstFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 19), .second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .second, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 37), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 11), .second, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .possible, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .possible, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .second, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .second]
set_option maxRecDepth 2048 in
theorem row9227_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9227 row9227_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9227_block_008
  · decide
  · rfl

def row9227_block_009 : List Nat := []
def plan9227_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, .first, (.secondFactor 17), .second, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9227_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9227 row9227_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9227_block_009
  · decide
  · rfl

def row9227_block_010 : List Nat := []
theorem row9227_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9227 row9227_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9227_block_011 : List Nat := []
theorem row9227_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9227 row9227_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9227_block_012 : List Nat := []
theorem row9227_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9227 row9227_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9227_block_013 : List Nat := []
theorem row9227_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9227 row9227_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9227_large_left_left : List Nat := row9227_small ++ row9227_block_005
theorem row9227_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9227 row9227_large_left_left :=
  Budget.RowUpperBound.branch row9227_small_covers row9227_block_005_covers

def row9227_large_left_right_right : List Nat := row9227_block_007 ++ row9227_block_008
theorem row9227_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9227 row9227_large_left_right_right :=
  Budget.RowUpperBound.branch row9227_block_007_covers row9227_block_008_covers

def row9227_large_left_right : List Nat := row9227_block_006 ++ row9227_large_left_right_right
theorem row9227_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9227 row9227_large_left_right :=
  Budget.RowUpperBound.branch row9227_block_006_covers row9227_large_left_right_right_covers

def row9227_large_left : List Nat := row9227_large_left_left ++ row9227_large_left_right
theorem row9227_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9227 row9227_large_left :=
  Budget.RowUpperBound.branch row9227_large_left_left_covers row9227_large_left_right_covers

def row9227_large_right_left : List Nat := row9227_block_009 ++ row9227_block_010
theorem row9227_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9227 row9227_large_right_left :=
  Budget.RowUpperBound.branch row9227_block_009_covers row9227_block_010_covers

def row9227_large_right_right_right : List Nat := row9227_block_012 ++ row9227_block_013
theorem row9227_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9227 row9227_large_right_right_right :=
  Budget.RowUpperBound.branch row9227_block_012_covers row9227_block_013_covers

def row9227_large_right_right : List Nat := row9227_block_011 ++ row9227_large_right_right_right
theorem row9227_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9227 row9227_large_right_right :=
  Budget.RowUpperBound.branch row9227_block_011_covers row9227_large_right_right_right_covers

def row9227_large_right : List Nat := row9227_large_right_left ++ row9227_large_right_right
theorem row9227_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9227 row9227_large_right :=
  Budget.RowUpperBound.branch row9227_large_right_left_covers row9227_large_right_right_covers

def row9227_large : List Nat := row9227_large_left ++ row9227_large_right
theorem row9227_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9227 row9227_large :=
  Budget.RowUpperBound.branch row9227_large_left_covers row9227_large_right_covers

def row9227 : List Nat := row9227_large
theorem row9227_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9227 row9227 := row9227_large_covers

def row9239_small : List Nat := []
theorem row9239_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9239 row9239_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9239_block_005 : List Nat := [9491, 9803, 9941, 10193, 10259]
def plan9239_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .possible, .first, .first, .first, (.secondFactor 13), .first, .second, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .possible, (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 13), (.firstFactor 7), .possible, .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9239_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9239 row9239_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9239_block_005
  · decide
  · rfl

def row9239_block_006 : List Nat := [11117, 11519, 11939]
def plan9239_block_006 : List Budget.Choice := [(.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .second, (.firstFactor 37), .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 29), (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.secondFactor 11), .first, .second, .second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 7), .second, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9239_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9239 row9239_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9239_block_006
  · decide
  · rfl

def row9239_block_007 : List Nat := [13121, 13499, 13577, 13997, 14447]
def plan9239_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.secondFactor 17), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 31), .second, .first, (.firstFactor 17), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row9239_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9239 row9239_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9239_block_007
  · decide
  · rfl

def row9239_block_008 : List Nat := [14627, 14633, 15173, 15287, 16421]
def plan9239_block_008 : List Budget.Choice := [.first, (.secondFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .possible, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .second, (.firstFactor 13), .possible, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .second, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9239_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9239 row9239_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9239_block_008
  · decide
  · rfl

def row9239_block_009 : List Nat := [16691]
def plan9239_block_009 : List Budget.Choice := [.first, (.firstFactor 31), .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9239_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9239 row9239_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9239_block_009
  · decide
  · rfl

def row9239_block_010 : List Nat := []
theorem row9239_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9239 row9239_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9239_block_011 : List Nat := []
theorem row9239_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9239 row9239_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9239_block_012 : List Nat := []
theorem row9239_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9239 row9239_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9239_block_013 : List Nat := []
theorem row9239_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9239 row9239_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9239_large_left_left : List Nat := row9239_small ++ row9239_block_005
theorem row9239_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9239 row9239_large_left_left :=
  Budget.RowUpperBound.branch row9239_small_covers row9239_block_005_covers

def row9239_large_left_right_right : List Nat := row9239_block_007 ++ row9239_block_008
theorem row9239_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9239 row9239_large_left_right_right :=
  Budget.RowUpperBound.branch row9239_block_007_covers row9239_block_008_covers

def row9239_large_left_right : List Nat := row9239_block_006 ++ row9239_large_left_right_right
theorem row9239_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9239 row9239_large_left_right :=
  Budget.RowUpperBound.branch row9239_block_006_covers row9239_large_left_right_right_covers

def row9239_large_left : List Nat := row9239_large_left_left ++ row9239_large_left_right
theorem row9239_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9239 row9239_large_left :=
  Budget.RowUpperBound.branch row9239_large_left_left_covers row9239_large_left_right_covers

def row9239_large_right_left : List Nat := row9239_block_009 ++ row9239_block_010
theorem row9239_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9239 row9239_large_right_left :=
  Budget.RowUpperBound.branch row9239_block_009_covers row9239_block_010_covers

def row9239_large_right_right_right : List Nat := row9239_block_012 ++ row9239_block_013
theorem row9239_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9239 row9239_large_right_right_right :=
  Budget.RowUpperBound.branch row9239_block_012_covers row9239_block_013_covers

def row9239_large_right_right : List Nat := row9239_block_011 ++ row9239_large_right_right_right
theorem row9239_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9239 row9239_large_right_right :=
  Budget.RowUpperBound.branch row9239_block_011_covers row9239_large_right_right_right_covers

def row9239_large_right : List Nat := row9239_large_right_left ++ row9239_large_right_right
theorem row9239_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9239 row9239_large_right :=
  Budget.RowUpperBound.branch row9239_large_right_left_covers row9239_large_right_right_covers

def row9239_large : List Nat := row9239_large_left ++ row9239_large_right
theorem row9239_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9239 row9239_large :=
  Budget.RowUpperBound.branch row9239_large_left_covers row9239_large_right_covers

def row9239 : List Nat := row9239_large
theorem row9239_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9239 row9239 := row9239_large_covers

def row9241_small : List Nat := []
theorem row9241_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9241 row9241_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9241_block_005 : List Nat := []
def plan9241_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9241_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9241 row9241_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9241_block_005
  · decide
  · rfl

def row9241_block_006 : List Nat := [10987, 11239, 11299]
def plan9241_block_006 : List Budget.Choice := [.first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 31), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .second, (.secondFactor 11), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .second, .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .possible, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 13), (.secondFactor 29), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 31), (.firstFactor 17), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .second, .second, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9241_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9241 row9241_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9241_block_006
  · decide
  · rfl

def row9241_block_007 : List Nat := [12547, 12799, 14449]
def plan9241_block_007 : List Budget.Choice := [.second, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .possible, .first, .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .possible, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row9241_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9241 row9241_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9241_block_007
  · decide
  · rfl

def row9241_block_008 : List Nat := [14767, 15583, 15679, 16111]
def plan9241_block_008 : List Budget.Choice := [(.firstFactor 37), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 29), (.firstFactor 31), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .possible, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 23), (.firstFactor 31), (.secondFactor 31), .first, .first, .second, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 37), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9241_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9241 row9241_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9241_block_008
  · decide
  · rfl

def row9241_block_009 : List Nat := []
def plan9241_block_009 : List Budget.Choice := [.second, .first, .first, .first, .second, (.firstFactor 11), .second, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9241_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9241 row9241_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9241_block_009
  · decide
  · rfl

def row9241_block_010 : List Nat := []
theorem row9241_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9241 row9241_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9241_block_011 : List Nat := []
theorem row9241_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9241 row9241_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9241_block_012 : List Nat := []
theorem row9241_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9241 row9241_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9241_block_013 : List Nat := []
theorem row9241_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9241 row9241_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9241_large_left_left : List Nat := row9241_small ++ row9241_block_005
theorem row9241_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9241 row9241_large_left_left :=
  Budget.RowUpperBound.branch row9241_small_covers row9241_block_005_covers

def row9241_large_left_right_right : List Nat := row9241_block_007 ++ row9241_block_008
theorem row9241_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9241 row9241_large_left_right_right :=
  Budget.RowUpperBound.branch row9241_block_007_covers row9241_block_008_covers

def row9241_large_left_right : List Nat := row9241_block_006 ++ row9241_large_left_right_right
theorem row9241_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9241 row9241_large_left_right :=
  Budget.RowUpperBound.branch row9241_block_006_covers row9241_large_left_right_right_covers

def row9241_large_left : List Nat := row9241_large_left_left ++ row9241_large_left_right
theorem row9241_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9241 row9241_large_left :=
  Budget.RowUpperBound.branch row9241_large_left_left_covers row9241_large_left_right_covers

def row9241_large_right_left : List Nat := row9241_block_009 ++ row9241_block_010
theorem row9241_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9241 row9241_large_right_left :=
  Budget.RowUpperBound.branch row9241_block_009_covers row9241_block_010_covers

def row9241_large_right_right_right : List Nat := row9241_block_012 ++ row9241_block_013
theorem row9241_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9241 row9241_large_right_right_right :=
  Budget.RowUpperBound.branch row9241_block_012_covers row9241_block_013_covers

def row9241_large_right_right : List Nat := row9241_block_011 ++ row9241_large_right_right_right
theorem row9241_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9241 row9241_large_right_right :=
  Budget.RowUpperBound.branch row9241_block_011_covers row9241_large_right_right_right_covers

def row9241_large_right : List Nat := row9241_large_right_left ++ row9241_large_right_right
theorem row9241_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9241 row9241_large_right :=
  Budget.RowUpperBound.branch row9241_large_right_left_covers row9241_large_right_right_covers

def row9241_large : List Nat := row9241_large_left ++ row9241_large_right
theorem row9241_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9241 row9241_large :=
  Budget.RowUpperBound.branch row9241_large_left_covers row9241_large_right_covers

def row9241 : List Nat := row9241_large
theorem row9241_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9241 row9241 := row9241_large_covers

end ProjectEulerSolutions.PrimePairLarge
