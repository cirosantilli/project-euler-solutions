import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10267_small : List Nat := []
theorem row10267_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10267 row10267_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10267_block_005 : List Nat := []
def plan10267_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row10267_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10267 row10267_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10267_block_005
  · decide
  · rfl

def row10267_block_006 : List Nat := [10837, 11251, 11593, 12097, 12409]
def plan10267_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 37), .first, .first, .second, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .possible, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 17), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .second, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 31), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row10267_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10267 row10267_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10267_block_006
  · decide
  · rfl

def row10267_block_007 : List Nat := [12919]
def plan10267_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .second, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .second, .first, .first, (.firstFactor 29), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10267_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10267 row10267_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10267_block_007
  · decide
  · rfl

def row10267_block_008 : List Nat := [14737, 15061, 15373]
def plan10267_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .possible, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 37), .first, .first, (.secondFactor 13), .first, (.secondFactor 29), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10267_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10267 row10267_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10267_block_008
  · decide
  · rfl

def row10267_block_009 : List Nat := []
theorem row10267_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10267 row10267_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10267_block_010 : List Nat := []
theorem row10267_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10267 row10267_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10267_block_011 : List Nat := []
theorem row10267_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10267 row10267_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10267_block_012 : List Nat := []
theorem row10267_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10267 row10267_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10267_block_013 : List Nat := []
theorem row10267_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10267 row10267_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10267_large_left_left : List Nat := row10267_small ++ row10267_block_005
theorem row10267_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10267 row10267_large_left_left :=
  Budget.RowUpperBound.branch row10267_small_covers row10267_block_005_covers

def row10267_large_left_right_right : List Nat := row10267_block_007 ++ row10267_block_008
theorem row10267_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10267 row10267_large_left_right_right :=
  Budget.RowUpperBound.branch row10267_block_007_covers row10267_block_008_covers

def row10267_large_left_right : List Nat := row10267_block_006 ++ row10267_large_left_right_right
theorem row10267_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10267 row10267_large_left_right :=
  Budget.RowUpperBound.branch row10267_block_006_covers row10267_large_left_right_right_covers

def row10267_large_left : List Nat := row10267_large_left_left ++ row10267_large_left_right
theorem row10267_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10267 row10267_large_left :=
  Budget.RowUpperBound.branch row10267_large_left_left_covers row10267_large_left_right_covers

def row10267_large_right_left : List Nat := row10267_block_009 ++ row10267_block_010
theorem row10267_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10267 row10267_large_right_left :=
  Budget.RowUpperBound.branch row10267_block_009_covers row10267_block_010_covers

def row10267_large_right_right_right : List Nat := row10267_block_012 ++ row10267_block_013
theorem row10267_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10267 row10267_large_right_right_right :=
  Budget.RowUpperBound.branch row10267_block_012_covers row10267_block_013_covers

def row10267_large_right_right : List Nat := row10267_block_011 ++ row10267_large_right_right_right
theorem row10267_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10267 row10267_large_right_right :=
  Budget.RowUpperBound.branch row10267_block_011_covers row10267_large_right_right_right_covers

def row10267_large_right : List Nat := row10267_large_right_left ++ row10267_large_right_right
theorem row10267_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10267 row10267_large_right :=
  Budget.RowUpperBound.branch row10267_large_right_left_covers row10267_large_right_right_covers

def row10267_large : List Nat := row10267_large_left ++ row10267_large_right
theorem row10267_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10267 row10267_large :=
  Budget.RowUpperBound.branch row10267_large_left_covers row10267_large_right_covers

def row10267 : List Nat := row10267_large
theorem row10267_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10267 row10267 := row10267_large_covers

def row10271_small : List Nat := []
theorem row10271_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10271 row10271_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10271_block_005 : List Nat := []
def plan10271_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .second, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10271_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10271 row10271_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10271_block_005
  · decide
  · rfl

def row10271_block_006 : List Nat := [10853, 11597]
def plan10271_block_006 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 31), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .second, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 37), (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second]
set_option maxRecDepth 2048 in
theorem row10271_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10271 row10271_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10271_block_006
  · decide
  · rfl

def row10271_block_007 : List Nat := [12641, 13127, 13337]
def plan10271_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .second, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .second, .possible, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 23), (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 37), .first, .second, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 37), .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row10271_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10271 row10271_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10271_block_007
  · decide
  · rfl

def row10271_block_008 : List Nat := []
def plan10271_block_008 : List Budget.Choice := [.first, (.firstFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, .second, (.secondFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 13), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10271_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10271 row10271_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10271_block_008
  · decide
  · rfl

def row10271_block_009 : List Nat := []
theorem row10271_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10271 row10271_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10271_block_010 : List Nat := []
theorem row10271_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10271 row10271_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10271_block_011 : List Nat := []
theorem row10271_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10271 row10271_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10271_block_012 : List Nat := []
theorem row10271_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10271 row10271_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10271_block_013 : List Nat := []
theorem row10271_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10271 row10271_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10271_large_left_left : List Nat := row10271_small ++ row10271_block_005
theorem row10271_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10271 row10271_large_left_left :=
  Budget.RowUpperBound.branch row10271_small_covers row10271_block_005_covers

def row10271_large_left_right_right : List Nat := row10271_block_007 ++ row10271_block_008
theorem row10271_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10271 row10271_large_left_right_right :=
  Budget.RowUpperBound.branch row10271_block_007_covers row10271_block_008_covers

def row10271_large_left_right : List Nat := row10271_block_006 ++ row10271_large_left_right_right
theorem row10271_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10271 row10271_large_left_right :=
  Budget.RowUpperBound.branch row10271_block_006_covers row10271_large_left_right_right_covers

def row10271_large_left : List Nat := row10271_large_left_left ++ row10271_large_left_right
theorem row10271_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10271 row10271_large_left :=
  Budget.RowUpperBound.branch row10271_large_left_left_covers row10271_large_left_right_covers

def row10271_large_right_left : List Nat := row10271_block_009 ++ row10271_block_010
theorem row10271_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10271 row10271_large_right_left :=
  Budget.RowUpperBound.branch row10271_block_009_covers row10271_block_010_covers

def row10271_large_right_right_right : List Nat := row10271_block_012 ++ row10271_block_013
theorem row10271_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10271 row10271_large_right_right_right :=
  Budget.RowUpperBound.branch row10271_block_012_covers row10271_block_013_covers

def row10271_large_right_right : List Nat := row10271_block_011 ++ row10271_large_right_right_right
theorem row10271_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10271 row10271_large_right_right :=
  Budget.RowUpperBound.branch row10271_block_011_covers row10271_large_right_right_right_covers

def row10271_large_right : List Nat := row10271_large_right_left ++ row10271_large_right_right
theorem row10271_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10271 row10271_large_right :=
  Budget.RowUpperBound.branch row10271_large_right_left_covers row10271_large_right_right_covers

def row10271_large : List Nat := row10271_large_left ++ row10271_large_right
theorem row10271_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10271 row10271_large :=
  Budget.RowUpperBound.branch row10271_large_left_covers row10271_large_right_covers

def row10271 : List Nat := row10271_large
theorem row10271_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10271 row10271 := row10271_large_covers

def row10273_small : List Nat := []
theorem row10273_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10273 row10273_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10273_block_005 : List Nat := [10303]
def plan10273_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .possible, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10273_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10273 row10273_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10273_block_005
  · decide
  · rfl

def row10273_block_006 : List Nat := [10987, 11731, 12109]
def plan10273_block_006 : List Budget.Choice := [.first, .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .possible, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), .second, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row10273_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10273 row10273_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10273_block_006
  · decide
  · rfl

def row10273_block_007 : List Nat := [13297]
def plan10273_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .second, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .second, .possible, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .second, .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row10273_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10273 row10273_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10273_block_007
  · decide
  · rfl

def row10273_block_008 : List Nat := [15277]
def plan10273_block_008 : List Budget.Choice := [(.secondFactor 31), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 13), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10273_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10273 row10273_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10273_block_008
  · decide
  · rfl

def row10273_block_009 : List Nat := []
theorem row10273_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10273 row10273_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10273_block_010 : List Nat := []
theorem row10273_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10273 row10273_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10273_block_011 : List Nat := []
theorem row10273_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10273 row10273_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10273_block_012 : List Nat := []
theorem row10273_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10273 row10273_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10273_block_013 : List Nat := []
theorem row10273_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10273 row10273_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10273_large_left_left : List Nat := row10273_small ++ row10273_block_005
theorem row10273_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10273 row10273_large_left_left :=
  Budget.RowUpperBound.branch row10273_small_covers row10273_block_005_covers

def row10273_large_left_right_right : List Nat := row10273_block_007 ++ row10273_block_008
theorem row10273_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10273 row10273_large_left_right_right :=
  Budget.RowUpperBound.branch row10273_block_007_covers row10273_block_008_covers

def row10273_large_left_right : List Nat := row10273_block_006 ++ row10273_large_left_right_right
theorem row10273_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10273 row10273_large_left_right :=
  Budget.RowUpperBound.branch row10273_block_006_covers row10273_large_left_right_right_covers

def row10273_large_left : List Nat := row10273_large_left_left ++ row10273_large_left_right
theorem row10273_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10273 row10273_large_left :=
  Budget.RowUpperBound.branch row10273_large_left_left_covers row10273_large_left_right_covers

def row10273_large_right_left : List Nat := row10273_block_009 ++ row10273_block_010
theorem row10273_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10273 row10273_large_right_left :=
  Budget.RowUpperBound.branch row10273_block_009_covers row10273_block_010_covers

def row10273_large_right_right_right : List Nat := row10273_block_012 ++ row10273_block_013
theorem row10273_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10273 row10273_large_right_right_right :=
  Budget.RowUpperBound.branch row10273_block_012_covers row10273_block_013_covers

def row10273_large_right_right : List Nat := row10273_block_011 ++ row10273_large_right_right_right
theorem row10273_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10273 row10273_large_right_right :=
  Budget.RowUpperBound.branch row10273_block_011_covers row10273_large_right_right_right_covers

def row10273_large_right : List Nat := row10273_large_right_left ++ row10273_large_right_right
theorem row10273_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10273 row10273_large_right :=
  Budget.RowUpperBound.branch row10273_large_right_left_covers row10273_large_right_right_covers

def row10273_large : List Nat := row10273_large_left ++ row10273_large_right
theorem row10273_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10273 row10273_large :=
  Budget.RowUpperBound.branch row10273_large_left_covers row10273_large_right_covers

def row10273 : List Nat := row10273_large
theorem row10273_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10273 row10273 := row10273_large_covers

def row10289_small : List Nat := []
theorem row10289_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10289 row10289_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10289_block_005 : List Nat := []
def plan10289_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row10289_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10289 row10289_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10289_block_005
  · decide
  · rfl

def row10289_block_006 : List Nat := [10427, 10859, 11069, 11489, 11657]
def plan10289_block_006 : List Budget.Choice := [(.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 29), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .possible, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 17), .possible, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 31), (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row10289_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10289 row10289_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10289_block_006
  · decide
  · rfl

def row10289_block_007 : List Nat := [12473, 13103, 13799, 14207, 14303]
def plan10289_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 29), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .possible, .first, .second, .first, (.secondFactor 29), (.firstFactor 19), .first, (.secondFactor 7), .first, .second, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row10289_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10289 row10289_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10289_block_007
  · decide
  · rfl

def row10289_block_008 : List Nat := []
def plan10289_block_008 : List Budget.Choice := [.first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .second, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .second, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 29), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10289_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10289 row10289_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10289_block_008
  · decide
  · rfl

def row10289_block_009 : List Nat := []
theorem row10289_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10289 row10289_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10289_block_010 : List Nat := []
theorem row10289_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10289 row10289_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10289_block_011 : List Nat := []
theorem row10289_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10289 row10289_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10289_block_012 : List Nat := []
theorem row10289_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10289 row10289_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10289_block_013 : List Nat := []
theorem row10289_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10289 row10289_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10289_large_left_left : List Nat := row10289_small ++ row10289_block_005
theorem row10289_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10289 row10289_large_left_left :=
  Budget.RowUpperBound.branch row10289_small_covers row10289_block_005_covers

def row10289_large_left_right_right : List Nat := row10289_block_007 ++ row10289_block_008
theorem row10289_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10289 row10289_large_left_right_right :=
  Budget.RowUpperBound.branch row10289_block_007_covers row10289_block_008_covers

def row10289_large_left_right : List Nat := row10289_block_006 ++ row10289_large_left_right_right
theorem row10289_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10289 row10289_large_left_right :=
  Budget.RowUpperBound.branch row10289_block_006_covers row10289_large_left_right_right_covers

def row10289_large_left : List Nat := row10289_large_left_left ++ row10289_large_left_right
theorem row10289_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10289 row10289_large_left :=
  Budget.RowUpperBound.branch row10289_large_left_left_covers row10289_large_left_right_covers

def row10289_large_right_left : List Nat := row10289_block_009 ++ row10289_block_010
theorem row10289_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10289 row10289_large_right_left :=
  Budget.RowUpperBound.branch row10289_block_009_covers row10289_block_010_covers

def row10289_large_right_right_right : List Nat := row10289_block_012 ++ row10289_block_013
theorem row10289_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10289 row10289_large_right_right_right :=
  Budget.RowUpperBound.branch row10289_block_012_covers row10289_block_013_covers

def row10289_large_right_right : List Nat := row10289_block_011 ++ row10289_large_right_right_right
theorem row10289_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10289 row10289_large_right_right :=
  Budget.RowUpperBound.branch row10289_block_011_covers row10289_large_right_right_right_covers

def row10289_large_right : List Nat := row10289_large_right_left ++ row10289_large_right_right
theorem row10289_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10289 row10289_large_right :=
  Budget.RowUpperBound.branch row10289_large_right_left_covers row10289_large_right_right_covers

def row10289_large : List Nat := row10289_large_left ++ row10289_large_right
theorem row10289_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10289 row10289_large :=
  Budget.RowUpperBound.branch row10289_large_left_covers row10289_large_right_covers

def row10289 : List Nat := row10289_large
theorem row10289_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10289 row10289 := row10289_large_covers

end ProjectEulerSolutions.PrimePairLarge
