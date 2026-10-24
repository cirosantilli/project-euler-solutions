import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11311_small : List Nat := []
theorem row11311_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11311 row11311_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11311_block_005 : List Nat := []
theorem row11311_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11311 row11311_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11311_block_006 : List Nat := []
def plan11311_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 37), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 17), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .second, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11311_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11311 row11311_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11311_block_006
  · decide
  · rfl

def row11311_block_007 : List Nat := [12487, 13099]
def plan11311_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, .first, .first, (.secondFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 31), (.firstFactor 13), .second, .first, .first, .second, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 37), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 23), (.firstFactor 17), .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .second, (.firstFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, (.firstFactor 23), (.secondFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row11311_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11311 row11311_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11311_block_007
  · decide
  · rfl

def row11311_block_008 : List Nat := []
def plan11311_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.secondFactor 37), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 31), .first, (.secondFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11311_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11311 row11311_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11311_block_008
  · decide
  · rfl

def row11311_block_009 : List Nat := []
theorem row11311_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11311 row11311_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11311_block_010 : List Nat := []
theorem row11311_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11311 row11311_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11311_block_011 : List Nat := []
theorem row11311_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11311 row11311_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11311_block_012 : List Nat := []
theorem row11311_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11311 row11311_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11311_block_013 : List Nat := []
theorem row11311_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11311 row11311_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11311_large_left_left : List Nat := row11311_small ++ row11311_block_005
theorem row11311_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11311 row11311_large_left_left :=
  Budget.RowUpperBound.branch row11311_small_covers row11311_block_005_covers

def row11311_large_left_right_right : List Nat := row11311_block_007 ++ row11311_block_008
theorem row11311_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11311 row11311_large_left_right_right :=
  Budget.RowUpperBound.branch row11311_block_007_covers row11311_block_008_covers

def row11311_large_left_right : List Nat := row11311_block_006 ++ row11311_large_left_right_right
theorem row11311_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11311 row11311_large_left_right :=
  Budget.RowUpperBound.branch row11311_block_006_covers row11311_large_left_right_right_covers

def row11311_large_left : List Nat := row11311_large_left_left ++ row11311_large_left_right
theorem row11311_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11311 row11311_large_left :=
  Budget.RowUpperBound.branch row11311_large_left_left_covers row11311_large_left_right_covers

def row11311_large_right_left : List Nat := row11311_block_009 ++ row11311_block_010
theorem row11311_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11311 row11311_large_right_left :=
  Budget.RowUpperBound.branch row11311_block_009_covers row11311_block_010_covers

def row11311_large_right_right_right : List Nat := row11311_block_012 ++ row11311_block_013
theorem row11311_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11311 row11311_large_right_right_right :=
  Budget.RowUpperBound.branch row11311_block_012_covers row11311_block_013_covers

def row11311_large_right_right : List Nat := row11311_block_011 ++ row11311_large_right_right_right
theorem row11311_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11311 row11311_large_right_right :=
  Budget.RowUpperBound.branch row11311_block_011_covers row11311_large_right_right_right_covers

def row11311_large_right : List Nat := row11311_large_right_left ++ row11311_large_right_right
theorem row11311_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11311 row11311_large_right :=
  Budget.RowUpperBound.branch row11311_large_right_left_covers row11311_large_right_right_covers

def row11311_large : List Nat := row11311_large_left ++ row11311_large_right
theorem row11311_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11311 row11311_large :=
  Budget.RowUpperBound.branch row11311_large_left_covers row11311_large_right_covers

def row11311 : List Nat := row11311_large
theorem row11311_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11311 row11311 := row11311_large_covers

def row11317_small : List Nat := []
theorem row11317_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11317 row11317_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11317_block_005 : List Nat := []
theorem row11317_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11317 row11317_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11317_block_006 : List Nat := [11551, 12157]
def plan11317_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, .possible, .first, .second, .second, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 29), .first, .second, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 29), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row11317_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11317 row11317_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11317_block_006
  · decide
  · rfl

def row11317_block_007 : List Nat := [12721]
def plan11317_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .second, (.secondFactor 37), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row11317_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11317 row11317_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11317_block_007
  · decide
  · rfl

def row11317_block_008 : List Nat := [14593]
def plan11317_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11317_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11317 row11317_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11317_block_008
  · decide
  · rfl

def row11317_block_009 : List Nat := []
theorem row11317_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11317 row11317_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11317_block_010 : List Nat := []
theorem row11317_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11317 row11317_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11317_block_011 : List Nat := []
theorem row11317_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11317 row11317_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11317_block_012 : List Nat := []
theorem row11317_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11317 row11317_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11317_block_013 : List Nat := []
theorem row11317_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11317 row11317_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11317_large_left_left : List Nat := row11317_small ++ row11317_block_005
theorem row11317_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11317 row11317_large_left_left :=
  Budget.RowUpperBound.branch row11317_small_covers row11317_block_005_covers

def row11317_large_left_right_right : List Nat := row11317_block_007 ++ row11317_block_008
theorem row11317_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11317 row11317_large_left_right_right :=
  Budget.RowUpperBound.branch row11317_block_007_covers row11317_block_008_covers

def row11317_large_left_right : List Nat := row11317_block_006 ++ row11317_large_left_right_right
theorem row11317_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11317 row11317_large_left_right :=
  Budget.RowUpperBound.branch row11317_block_006_covers row11317_large_left_right_right_covers

def row11317_large_left : List Nat := row11317_large_left_left ++ row11317_large_left_right
theorem row11317_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11317 row11317_large_left :=
  Budget.RowUpperBound.branch row11317_large_left_left_covers row11317_large_left_right_covers

def row11317_large_right_left : List Nat := row11317_block_009 ++ row11317_block_010
theorem row11317_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11317 row11317_large_right_left :=
  Budget.RowUpperBound.branch row11317_block_009_covers row11317_block_010_covers

def row11317_large_right_right_right : List Nat := row11317_block_012 ++ row11317_block_013
theorem row11317_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11317 row11317_large_right_right_right :=
  Budget.RowUpperBound.branch row11317_block_012_covers row11317_block_013_covers

def row11317_large_right_right : List Nat := row11317_block_011 ++ row11317_large_right_right_right
theorem row11317_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11317 row11317_large_right_right :=
  Budget.RowUpperBound.branch row11317_block_011_covers row11317_large_right_right_right_covers

def row11317_large_right : List Nat := row11317_large_right_left ++ row11317_large_right_right
theorem row11317_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11317 row11317_large_right :=
  Budget.RowUpperBound.branch row11317_large_right_left_covers row11317_large_right_right_covers

def row11317_large : List Nat := row11317_large_left ++ row11317_large_right
theorem row11317_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11317 row11317_large :=
  Budget.RowUpperBound.branch row11317_large_left_covers row11317_large_right_covers

def row11317 : List Nat := row11317_large
theorem row11317_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11317 row11317 := row11317_large_covers

def row11321_small : List Nat := []
theorem row11321_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11321 row11321_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11321_block_005 : List Nat := []
theorem row11321_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11321 row11321_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11321_block_006 : List Nat := [11987]
def plan11321_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 31), (.firstFactor 29), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.secondFactor 13), .second, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row11321_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11321 row11321_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11321_block_006
  · decide
  · rfl

def row11321_block_007 : List Nat := [12497, 13883, 13907]
def plan11321_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.firstFactor 19), (.secondFactor 37), .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 37), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .second, .second, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .second, .first, .possible, (.firstFactor 37), .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row11321_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11321 row11321_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11321_block_007
  · decide
  · rfl

def row11321_block_008 : List Nat := []
def plan11321_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11321_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11321 row11321_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11321_block_008
  · decide
  · rfl

def row11321_block_009 : List Nat := []
theorem row11321_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11321 row11321_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11321_block_010 : List Nat := []
theorem row11321_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11321 row11321_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11321_block_011 : List Nat := []
theorem row11321_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11321 row11321_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11321_block_012 : List Nat := []
theorem row11321_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11321 row11321_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11321_block_013 : List Nat := []
theorem row11321_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11321 row11321_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11321_large_left_left : List Nat := row11321_small ++ row11321_block_005
theorem row11321_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11321 row11321_large_left_left :=
  Budget.RowUpperBound.branch row11321_small_covers row11321_block_005_covers

def row11321_large_left_right_right : List Nat := row11321_block_007 ++ row11321_block_008
theorem row11321_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11321 row11321_large_left_right_right :=
  Budget.RowUpperBound.branch row11321_block_007_covers row11321_block_008_covers

def row11321_large_left_right : List Nat := row11321_block_006 ++ row11321_large_left_right_right
theorem row11321_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11321 row11321_large_left_right :=
  Budget.RowUpperBound.branch row11321_block_006_covers row11321_large_left_right_right_covers

def row11321_large_left : List Nat := row11321_large_left_left ++ row11321_large_left_right
theorem row11321_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11321 row11321_large_left :=
  Budget.RowUpperBound.branch row11321_large_left_left_covers row11321_large_left_right_covers

def row11321_large_right_left : List Nat := row11321_block_009 ++ row11321_block_010
theorem row11321_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11321 row11321_large_right_left :=
  Budget.RowUpperBound.branch row11321_block_009_covers row11321_block_010_covers

def row11321_large_right_right_right : List Nat := row11321_block_012 ++ row11321_block_013
theorem row11321_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11321 row11321_large_right_right_right :=
  Budget.RowUpperBound.branch row11321_block_012_covers row11321_block_013_covers

def row11321_large_right_right : List Nat := row11321_block_011 ++ row11321_large_right_right_right
theorem row11321_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11321 row11321_large_right_right :=
  Budget.RowUpperBound.branch row11321_block_011_covers row11321_large_right_right_right_covers

def row11321_large_right : List Nat := row11321_large_right_left ++ row11321_large_right_right
theorem row11321_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11321 row11321_large_right :=
  Budget.RowUpperBound.branch row11321_large_right_left_covers row11321_large_right_right_covers

def row11321_large : List Nat := row11321_large_left ++ row11321_large_right
theorem row11321_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11321 row11321_large :=
  Budget.RowUpperBound.branch row11321_large_left_covers row11321_large_right_covers

def row11321 : List Nat := row11321_large
theorem row11321_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11321 row11321 := row11321_large_covers

def row11329_small : List Nat := []
theorem row11329_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11329 row11329_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11329_block_005 : List Nat := []
theorem row11329_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11329 row11329_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11329_block_006 : List Nat := [12097]
def plan11329_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .second, .first, .second, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11329_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11329 row11329_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11329_block_006
  · decide
  · rfl

def row11329_block_007 : List Nat := [12421, 13003, 13147, 13381, 13633, 14389]
def plan11329_block_007 : List Budget.Choice := [.possible, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 31), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 29), .second, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 19), .possible, .first, (.secondFactor 7), .first, (.firstFactor 29), .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .second, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11329_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11329 row11329_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11329_block_007
  · decide
  · rfl

def row11329_block_008 : List Nat := []
def plan11329_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .second, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11329_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11329 row11329_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11329_block_008
  · decide
  · rfl

def row11329_block_009 : List Nat := []
theorem row11329_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11329 row11329_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11329_block_010 : List Nat := []
theorem row11329_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11329 row11329_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11329_block_011 : List Nat := []
theorem row11329_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11329 row11329_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11329_block_012 : List Nat := []
theorem row11329_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11329 row11329_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11329_block_013 : List Nat := []
theorem row11329_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11329 row11329_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11329_large_left_left : List Nat := row11329_small ++ row11329_block_005
theorem row11329_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11329 row11329_large_left_left :=
  Budget.RowUpperBound.branch row11329_small_covers row11329_block_005_covers

def row11329_large_left_right_right : List Nat := row11329_block_007 ++ row11329_block_008
theorem row11329_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11329 row11329_large_left_right_right :=
  Budget.RowUpperBound.branch row11329_block_007_covers row11329_block_008_covers

def row11329_large_left_right : List Nat := row11329_block_006 ++ row11329_large_left_right_right
theorem row11329_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11329 row11329_large_left_right :=
  Budget.RowUpperBound.branch row11329_block_006_covers row11329_large_left_right_right_covers

def row11329_large_left : List Nat := row11329_large_left_left ++ row11329_large_left_right
theorem row11329_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11329 row11329_large_left :=
  Budget.RowUpperBound.branch row11329_large_left_left_covers row11329_large_left_right_covers

def row11329_large_right_left : List Nat := row11329_block_009 ++ row11329_block_010
theorem row11329_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11329 row11329_large_right_left :=
  Budget.RowUpperBound.branch row11329_block_009_covers row11329_block_010_covers

def row11329_large_right_right_right : List Nat := row11329_block_012 ++ row11329_block_013
theorem row11329_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11329 row11329_large_right_right_right :=
  Budget.RowUpperBound.branch row11329_block_012_covers row11329_block_013_covers

def row11329_large_right_right : List Nat := row11329_block_011 ++ row11329_large_right_right_right
theorem row11329_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11329 row11329_large_right_right :=
  Budget.RowUpperBound.branch row11329_block_011_covers row11329_large_right_right_right_covers

def row11329_large_right : List Nat := row11329_large_right_left ++ row11329_large_right_right
theorem row11329_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11329 row11329_large_right :=
  Budget.RowUpperBound.branch row11329_large_right_left_covers row11329_large_right_right_covers

def row11329_large : List Nat := row11329_large_left ++ row11329_large_right
theorem row11329_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11329 row11329_large :=
  Budget.RowUpperBound.branch row11329_large_left_covers row11329_large_right_covers

def row11329 : List Nat := row11329_large
theorem row11329_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11329 row11329 := row11329_large_covers

end ProjectEulerSolutions.PrimePairLarge
