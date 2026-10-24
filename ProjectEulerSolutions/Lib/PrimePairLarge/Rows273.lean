import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8783_small : List Nat := []
theorem row8783_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8783 row8783_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8783_block_005 : List Nat := [9281, 9473, 9551, 9767, 9803]
def plan8783_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .second, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.secondFactor 29), .possible, .first, .first, .first, (.firstFactor 13), .possible, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 17), (.firstFactor 31), .first, (.secondFactor 29), .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8783_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8783 row8783_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8783_block_005
  · decide
  · rfl

def row8783_block_006 : List Nat := [11657, 12143, 12269]
def plan8783_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 31), (.secondFactor 11), .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .second, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .possible, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 13), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 29), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 31), .possible, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 23), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8783_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8783 row8783_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8783_block_006
  · decide
  · rfl

def row8783_block_007 : List Nat := [13463]
def plan8783_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 29), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .possible, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .second, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row8783_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8783 row8783_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8783_block_007
  · decide
  · rfl

def row8783_block_008 : List Nat := [14699, 15077, 16427]
def plan8783_block_008 : List Budget.Choice := [.first, (.firstFactor 13), .first, .second, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 29), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.secondFactor 11), (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .second, .first, .second, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8783_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8783 row8783_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8783_block_008
  · decide
  · rfl

def row8783_block_009 : List Nat := [17207]
def plan8783_block_009 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .second, (.secondFactor 7), (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8783_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8783 row8783_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8783_block_009
  · decide
  · rfl

def row8783_block_010 : List Nat := []
theorem row8783_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8783 row8783_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8783_block_011 : List Nat := []
theorem row8783_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8783 row8783_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8783_block_012 : List Nat := []
theorem row8783_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8783 row8783_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8783_block_013 : List Nat := []
theorem row8783_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8783 row8783_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8783_large_left_left : List Nat := row8783_small ++ row8783_block_005
theorem row8783_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8783 row8783_large_left_left :=
  Budget.RowUpperBound.branch row8783_small_covers row8783_block_005_covers

def row8783_large_left_right_right : List Nat := row8783_block_007 ++ row8783_block_008
theorem row8783_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8783 row8783_large_left_right_right :=
  Budget.RowUpperBound.branch row8783_block_007_covers row8783_block_008_covers

def row8783_large_left_right : List Nat := row8783_block_006 ++ row8783_large_left_right_right
theorem row8783_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8783 row8783_large_left_right :=
  Budget.RowUpperBound.branch row8783_block_006_covers row8783_large_left_right_right_covers

def row8783_large_left : List Nat := row8783_large_left_left ++ row8783_large_left_right
theorem row8783_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8783 row8783_large_left :=
  Budget.RowUpperBound.branch row8783_large_left_left_covers row8783_large_left_right_covers

def row8783_large_right_left : List Nat := row8783_block_009 ++ row8783_block_010
theorem row8783_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8783 row8783_large_right_left :=
  Budget.RowUpperBound.branch row8783_block_009_covers row8783_block_010_covers

def row8783_large_right_right_right : List Nat := row8783_block_012 ++ row8783_block_013
theorem row8783_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8783 row8783_large_right_right_right :=
  Budget.RowUpperBound.branch row8783_block_012_covers row8783_block_013_covers

def row8783_large_right_right : List Nat := row8783_block_011 ++ row8783_large_right_right_right
theorem row8783_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8783 row8783_large_right_right :=
  Budget.RowUpperBound.branch row8783_block_011_covers row8783_large_right_right_right_covers

def row8783_large_right : List Nat := row8783_large_right_left ++ row8783_large_right_right
theorem row8783_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8783 row8783_large_right :=
  Budget.RowUpperBound.branch row8783_large_right_left_covers row8783_large_right_right_covers

def row8783_large : List Nat := row8783_large_left ++ row8783_large_right
theorem row8783_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8783 row8783_large :=
  Budget.RowUpperBound.branch row8783_large_left_covers row8783_large_right_covers

def row8783 : List Nat := row8783_large
theorem row8783_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8783 row8783 := row8783_large_covers

def row8803_small : List Nat := []
theorem row8803_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8803 row8803_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8803_block_005 : List Nat := [9871]
def plan8803_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .second, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .second, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .second, (.secondFactor 23), .first, .first, .first, .first, .first, .possible, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row8803_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8803 row8803_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8803_block_005
  · decide
  · rfl

def row8803_block_006 : List Nat := [10531, 11491, 11827, 12049, 12409]
def plan8803_block_006 : List Budget.Choice := [.first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), .possible, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 13), .possible, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .second, (.firstFactor 7), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row8803_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8803 row8803_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8803_block_006
  · decide
  · rfl

def row8803_block_007 : List Nat := [12967, 13933]
def plan8803_block_007 : List Budget.Choice := [(.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), (.secondFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, (.secondFactor 13), (.firstFactor 29), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8803_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8803 row8803_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8803_block_007
  · decide
  · rfl

def row8803_block_008 : List Nat := [14947, 15451]
def plan8803_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .possible, .first, .first, .first, .second, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 31), .first, (.firstFactor 31), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 23), (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 23), (.secondFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8803_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8803 row8803_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8803_block_008
  · decide
  · rfl

def row8803_block_009 : List Nat := [16693, 16747]
def plan8803_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 17), .first, (.firstFactor 17), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 37), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8803_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8803 row8803_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8803_block_009
  · decide
  · rfl

def row8803_block_010 : List Nat := []
theorem row8803_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8803 row8803_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8803_block_011 : List Nat := []
theorem row8803_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8803 row8803_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8803_block_012 : List Nat := []
theorem row8803_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8803 row8803_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8803_block_013 : List Nat := []
theorem row8803_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8803 row8803_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8803_large_left_left : List Nat := row8803_small ++ row8803_block_005
theorem row8803_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8803 row8803_large_left_left :=
  Budget.RowUpperBound.branch row8803_small_covers row8803_block_005_covers

def row8803_large_left_right_right : List Nat := row8803_block_007 ++ row8803_block_008
theorem row8803_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8803 row8803_large_left_right_right :=
  Budget.RowUpperBound.branch row8803_block_007_covers row8803_block_008_covers

def row8803_large_left_right : List Nat := row8803_block_006 ++ row8803_large_left_right_right
theorem row8803_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8803 row8803_large_left_right :=
  Budget.RowUpperBound.branch row8803_block_006_covers row8803_large_left_right_right_covers

def row8803_large_left : List Nat := row8803_large_left_left ++ row8803_large_left_right
theorem row8803_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8803 row8803_large_left :=
  Budget.RowUpperBound.branch row8803_large_left_left_covers row8803_large_left_right_covers

def row8803_large_right_left : List Nat := row8803_block_009 ++ row8803_block_010
theorem row8803_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8803 row8803_large_right_left :=
  Budget.RowUpperBound.branch row8803_block_009_covers row8803_block_010_covers

def row8803_large_right_right_right : List Nat := row8803_block_012 ++ row8803_block_013
theorem row8803_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8803 row8803_large_right_right_right :=
  Budget.RowUpperBound.branch row8803_block_012_covers row8803_block_013_covers

def row8803_large_right_right : List Nat := row8803_block_011 ++ row8803_large_right_right_right
theorem row8803_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8803 row8803_large_right_right :=
  Budget.RowUpperBound.branch row8803_block_011_covers row8803_large_right_right_right_covers

def row8803_large_right : List Nat := row8803_large_right_left ++ row8803_large_right_right
theorem row8803_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8803 row8803_large_right :=
  Budget.RowUpperBound.branch row8803_large_right_left_covers row8803_large_right_right_covers

def row8803_large : List Nat := row8803_large_left ++ row8803_large_right
theorem row8803_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8803 row8803_large :=
  Budget.RowUpperBound.branch row8803_large_left_covers row8803_large_right_covers

def row8803 : List Nat := row8803_large
theorem row8803_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8803 row8803 := row8803_large_covers

def row8807_small : List Nat := []
theorem row8807_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8807 row8807_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8807_block_005 : List Nat := [9209, 9239, 10139, 10301]
def plan8807_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .possible, .first, (.firstFactor 17), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .second, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), (.firstFactor 37), .first, .second, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8807_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8807 row8807_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8807_block_005
  · decide
  · rfl

def row8807_block_006 : List Nat := [11369, 11801, 11927]
def plan8807_block_006 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 23), .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 11), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, .second, (.secondFactor 13), (.firstFactor 29), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .possible, (.firstFactor 19), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .possible, (.firstFactor 29), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8807_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8807 row8807_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8807_block_006
  · decide
  · rfl

def row8807_block_007 : List Nat := [13901]
def plan8807_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 17), .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 13), (.firstFactor 7), .second, .first, .first, .first, .second, (.firstFactor 37), (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 23), .first, (.firstFactor 11), .second, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8807_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8807 row8807_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8807_block_007
  · decide
  · rfl

def row8807_block_008 : List Nat := []
def plan8807_block_008 : List Budget.Choice := [.first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .second, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .second, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 11), (.firstFactor 7), .first, .second, (.firstFactor 13), (.firstFactor 11), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.secondFactor 11), (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 37), .second, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row8807_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8807 row8807_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8807_block_008
  · decide
  · rfl

def row8807_block_009 : List Nat := [16787, 16871]
def plan8807_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .possible, .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8807_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8807 row8807_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8807_block_009
  · decide
  · rfl

def row8807_block_010 : List Nat := []
theorem row8807_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8807 row8807_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8807_block_011 : List Nat := []
theorem row8807_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8807 row8807_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8807_block_012 : List Nat := []
theorem row8807_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8807 row8807_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8807_block_013 : List Nat := []
theorem row8807_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8807 row8807_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8807_large_left_left : List Nat := row8807_small ++ row8807_block_005
theorem row8807_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8807 row8807_large_left_left :=
  Budget.RowUpperBound.branch row8807_small_covers row8807_block_005_covers

def row8807_large_left_right_right : List Nat := row8807_block_007 ++ row8807_block_008
theorem row8807_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8807 row8807_large_left_right_right :=
  Budget.RowUpperBound.branch row8807_block_007_covers row8807_block_008_covers

def row8807_large_left_right : List Nat := row8807_block_006 ++ row8807_large_left_right_right
theorem row8807_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8807 row8807_large_left_right :=
  Budget.RowUpperBound.branch row8807_block_006_covers row8807_large_left_right_right_covers

def row8807_large_left : List Nat := row8807_large_left_left ++ row8807_large_left_right
theorem row8807_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8807 row8807_large_left :=
  Budget.RowUpperBound.branch row8807_large_left_left_covers row8807_large_left_right_covers

def row8807_large_right_left : List Nat := row8807_block_009 ++ row8807_block_010
theorem row8807_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8807 row8807_large_right_left :=
  Budget.RowUpperBound.branch row8807_block_009_covers row8807_block_010_covers

def row8807_large_right_right_right : List Nat := row8807_block_012 ++ row8807_block_013
theorem row8807_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8807 row8807_large_right_right_right :=
  Budget.RowUpperBound.branch row8807_block_012_covers row8807_block_013_covers

def row8807_large_right_right : List Nat := row8807_block_011 ++ row8807_large_right_right_right
theorem row8807_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8807 row8807_large_right_right :=
  Budget.RowUpperBound.branch row8807_block_011_covers row8807_large_right_right_right_covers

def row8807_large_right : List Nat := row8807_large_right_left ++ row8807_large_right_right
theorem row8807_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8807 row8807_large_right :=
  Budget.RowUpperBound.branch row8807_large_right_left_covers row8807_large_right_right_covers

def row8807_large : List Nat := row8807_large_left ++ row8807_large_right
theorem row8807_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8807 row8807_large :=
  Budget.RowUpperBound.branch row8807_large_left_covers row8807_large_right_covers

def row8807 : List Nat := row8807_large
theorem row8807_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8807 row8807 := row8807_large_covers

def row8819_small : List Nat := []
theorem row8819_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8819 row8819_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8819_block_005 : List Nat := [9239, 9719, 10181]
def plan8819_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .possible, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .possible, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), .first, .possible, (.secondFactor 19), (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8819_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8819 row8819_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8819_block_005
  · decide
  · rfl

def row8819_block_006 : List Nat := [11171, 11411]
def plan8819_block_006 : List Budget.Choice := [.second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .second, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 19), (.secondFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 11), .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 17), .first, .second, (.firstFactor 11), .first, (.firstFactor 29), .first, .second, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .second, .first, (.secondFactor 17), .first, .first, (.secondFactor 29), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8819_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8819 row8819_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8819_block_006
  · decide
  · rfl

def row8819_block_007 : List Nat := [13829, 13931, 13997, 14057]
def plan8819_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.secondFactor 23), (.firstFactor 37), .second, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 19), .possible, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row8819_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8819 row8819_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8819_block_007
  · decide
  · rfl

def row8819_block_008 : List Nat := [15383]
def plan8819_block_008 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, (.firstFactor 23), .first, (.firstFactor 31), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 7), .first, .second, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 13), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row8819_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8819 row8819_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8819_block_008
  · decide
  · rfl

def row8819_block_009 : List Nat := [16619, 17123]
def plan8819_block_009 : List Budget.Choice := [.first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible, (.secondFactor 23), .first, .second, .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 11), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8819_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8819 row8819_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8819_block_009
  · decide
  · rfl

def row8819_block_010 : List Nat := []
theorem row8819_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8819 row8819_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8819_block_011 : List Nat := []
theorem row8819_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8819 row8819_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8819_block_012 : List Nat := []
theorem row8819_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8819 row8819_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8819_block_013 : List Nat := []
theorem row8819_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8819 row8819_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8819_large_left_left : List Nat := row8819_small ++ row8819_block_005
theorem row8819_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8819 row8819_large_left_left :=
  Budget.RowUpperBound.branch row8819_small_covers row8819_block_005_covers

def row8819_large_left_right_right : List Nat := row8819_block_007 ++ row8819_block_008
theorem row8819_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8819 row8819_large_left_right_right :=
  Budget.RowUpperBound.branch row8819_block_007_covers row8819_block_008_covers

def row8819_large_left_right : List Nat := row8819_block_006 ++ row8819_large_left_right_right
theorem row8819_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8819 row8819_large_left_right :=
  Budget.RowUpperBound.branch row8819_block_006_covers row8819_large_left_right_right_covers

def row8819_large_left : List Nat := row8819_large_left_left ++ row8819_large_left_right
theorem row8819_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8819 row8819_large_left :=
  Budget.RowUpperBound.branch row8819_large_left_left_covers row8819_large_left_right_covers

def row8819_large_right_left : List Nat := row8819_block_009 ++ row8819_block_010
theorem row8819_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8819 row8819_large_right_left :=
  Budget.RowUpperBound.branch row8819_block_009_covers row8819_block_010_covers

def row8819_large_right_right_right : List Nat := row8819_block_012 ++ row8819_block_013
theorem row8819_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8819 row8819_large_right_right_right :=
  Budget.RowUpperBound.branch row8819_block_012_covers row8819_block_013_covers

def row8819_large_right_right : List Nat := row8819_block_011 ++ row8819_large_right_right_right
theorem row8819_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8819 row8819_large_right_right :=
  Budget.RowUpperBound.branch row8819_block_011_covers row8819_large_right_right_right_covers

def row8819_large_right : List Nat := row8819_large_right_left ++ row8819_large_right_right
theorem row8819_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8819 row8819_large_right :=
  Budget.RowUpperBound.branch row8819_large_right_left_covers row8819_large_right_right_covers

def row8819_large : List Nat := row8819_large_left ++ row8819_large_right
theorem row8819_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8819 row8819_large :=
  Budget.RowUpperBound.branch row8819_large_left_covers row8819_large_right_covers

def row8819 : List Nat := row8819_large
theorem row8819_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8819 row8819 := row8819_large_covers

end ProjectEulerSolutions.PrimePairLarge
