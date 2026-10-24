import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10883_small : List Nat := []
theorem row10883_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10883 row10883_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10883_block_005 : List Nat := []
theorem row10883_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10883 row10883_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10883_block_006 : List Nat := [11519, 12227]
def plan10883_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, (.secondFactor 31), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 29), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row10883_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10883 row10883_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10883_block_006
  · decide
  · rfl

def row10883_block_007 : List Nat := [13337, 13451, 13619, 13841]
def plan10883_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, .second, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row10883_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10883 row10883_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10883_block_007
  · decide
  · rfl

def row10883_block_008 : List Nat := [14831, 14957]
def plan10883_block_008 : List Budget.Choice := [.first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .second, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 37), .first, .second, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, (.secondFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10883_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10883 row10883_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10883_block_008
  · decide
  · rfl

def row10883_block_009 : List Nat := []
theorem row10883_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10883 row10883_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10883_block_010 : List Nat := []
theorem row10883_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10883 row10883_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10883_block_011 : List Nat := []
theorem row10883_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10883 row10883_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10883_block_012 : List Nat := []
theorem row10883_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10883 row10883_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10883_block_013 : List Nat := []
theorem row10883_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10883 row10883_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10883_large_left_left : List Nat := row10883_small ++ row10883_block_005
theorem row10883_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10883 row10883_large_left_left :=
  Budget.RowUpperBound.branch row10883_small_covers row10883_block_005_covers

def row10883_large_left_right_right : List Nat := row10883_block_007 ++ row10883_block_008
theorem row10883_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10883 row10883_large_left_right_right :=
  Budget.RowUpperBound.branch row10883_block_007_covers row10883_block_008_covers

def row10883_large_left_right : List Nat := row10883_block_006 ++ row10883_large_left_right_right
theorem row10883_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10883 row10883_large_left_right :=
  Budget.RowUpperBound.branch row10883_block_006_covers row10883_large_left_right_right_covers

def row10883_large_left : List Nat := row10883_large_left_left ++ row10883_large_left_right
theorem row10883_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10883 row10883_large_left :=
  Budget.RowUpperBound.branch row10883_large_left_left_covers row10883_large_left_right_covers

def row10883_large_right_left : List Nat := row10883_block_009 ++ row10883_block_010
theorem row10883_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10883 row10883_large_right_left :=
  Budget.RowUpperBound.branch row10883_block_009_covers row10883_block_010_covers

def row10883_large_right_right_right : List Nat := row10883_block_012 ++ row10883_block_013
theorem row10883_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10883 row10883_large_right_right_right :=
  Budget.RowUpperBound.branch row10883_block_012_covers row10883_block_013_covers

def row10883_large_right_right : List Nat := row10883_block_011 ++ row10883_large_right_right_right
theorem row10883_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10883 row10883_large_right_right :=
  Budget.RowUpperBound.branch row10883_block_011_covers row10883_large_right_right_right_covers

def row10883_large_right : List Nat := row10883_large_right_left ++ row10883_large_right_right
theorem row10883_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10883 row10883_large_right :=
  Budget.RowUpperBound.branch row10883_large_right_left_covers row10883_large_right_right_covers

def row10883_large : List Nat := row10883_large_left ++ row10883_large_right
theorem row10883_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10883 row10883_large :=
  Budget.RowUpperBound.branch row10883_large_left_covers row10883_large_right_covers

def row10883 : List Nat := row10883_large
theorem row10883_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10883 row10883 := row10883_large_covers

def row10889_small : List Nat := []
theorem row10889_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10889 row10889_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10889_block_005 : List Nat := []
theorem row10889_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10889 row10889_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10889_block_006 : List Nat := [11003, 11903, 11927]
def plan10889_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 13), (.firstFactor 7), .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .possible, (.secondFactor 7), .first, .possible, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 31), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .second, (.firstFactor 29), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10889_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10889 row10889_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10889_block_006
  · decide
  · rfl

def row10889_block_007 : List Nat := [12821, 12911, 13109, 13229, 13577, 13877, 14081, 14087]
def plan10889_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, .second, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .possible, .first, .first, .first, .first, .first, (.secondFactor 17), .second, .first, .possible, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .possible, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .second, .second, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .possible, .first, (.firstFactor 19), (.secondFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, .possible, .first, .first, .first, (.firstFactor 31), (.secondFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row10889_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10889 row10889_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10889_block_007
  · decide
  · rfl

def row10889_block_008 : List Nat := [14543]
def plan10889_block_008 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .second, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, .second, (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10889_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10889 row10889_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10889_block_008
  · decide
  · rfl

def row10889_block_009 : List Nat := []
theorem row10889_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10889 row10889_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10889_block_010 : List Nat := []
theorem row10889_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10889 row10889_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10889_block_011 : List Nat := []
theorem row10889_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10889 row10889_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10889_block_012 : List Nat := []
theorem row10889_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10889 row10889_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10889_block_013 : List Nat := []
theorem row10889_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10889 row10889_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10889_large_left_left : List Nat := row10889_small ++ row10889_block_005
theorem row10889_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10889 row10889_large_left_left :=
  Budget.RowUpperBound.branch row10889_small_covers row10889_block_005_covers

def row10889_large_left_right_right : List Nat := row10889_block_007 ++ row10889_block_008
theorem row10889_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10889 row10889_large_left_right_right :=
  Budget.RowUpperBound.branch row10889_block_007_covers row10889_block_008_covers

def row10889_large_left_right : List Nat := row10889_block_006 ++ row10889_large_left_right_right
theorem row10889_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10889 row10889_large_left_right :=
  Budget.RowUpperBound.branch row10889_block_006_covers row10889_large_left_right_right_covers

def row10889_large_left : List Nat := row10889_large_left_left ++ row10889_large_left_right
theorem row10889_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10889 row10889_large_left :=
  Budget.RowUpperBound.branch row10889_large_left_left_covers row10889_large_left_right_covers

def row10889_large_right_left : List Nat := row10889_block_009 ++ row10889_block_010
theorem row10889_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10889 row10889_large_right_left :=
  Budget.RowUpperBound.branch row10889_block_009_covers row10889_block_010_covers

def row10889_large_right_right_right : List Nat := row10889_block_012 ++ row10889_block_013
theorem row10889_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10889 row10889_large_right_right_right :=
  Budget.RowUpperBound.branch row10889_block_012_covers row10889_block_013_covers

def row10889_large_right_right : List Nat := row10889_block_011 ++ row10889_large_right_right_right
theorem row10889_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10889 row10889_large_right_right :=
  Budget.RowUpperBound.branch row10889_block_011_covers row10889_large_right_right_right_covers

def row10889_large_right : List Nat := row10889_large_right_left ++ row10889_large_right_right
theorem row10889_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10889 row10889_large_right :=
  Budget.RowUpperBound.branch row10889_large_right_left_covers row10889_large_right_right_covers

def row10889_large : List Nat := row10889_large_left ++ row10889_large_right
theorem row10889_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10889 row10889_large :=
  Budget.RowUpperBound.branch row10889_large_left_covers row10889_large_right_covers

def row10889 : List Nat := row10889_large
theorem row10889_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10889 row10889 := row10889_large_covers

def row10891_small : List Nat := []
theorem row10891_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10891 row10891_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10891_block_005 : List Nat := []
theorem row10891_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10891 row10891_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10891_block_006 : List Nat := [11299, 11701, 11719, 12163]
def plan10891_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .second, .first, .first, .second, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, .possible, .first, .possible, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10891_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10891 row10891_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10891_block_006
  · decide
  · rfl

def row10891_block_007 : List Nat := [12511, 12781, 14419, 14431]
def plan10891_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 29), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .second, .first, .first, .first, (.secondFactor 31), (.firstFactor 37), .second, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 31), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .second, (.secondFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 19), .first, .possible, .first, .possible, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10891_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10891 row10891_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10891_block_007
  · decide
  · rfl

def row10891_block_008 : List Nat := []
def plan10891_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .second, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10891_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10891 row10891_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10891_block_008
  · decide
  · rfl

def row10891_block_009 : List Nat := []
theorem row10891_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10891 row10891_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10891_block_010 : List Nat := []
theorem row10891_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10891 row10891_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10891_block_011 : List Nat := []
theorem row10891_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10891 row10891_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10891_block_012 : List Nat := []
theorem row10891_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10891 row10891_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10891_block_013 : List Nat := []
theorem row10891_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10891 row10891_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10891_large_left_left : List Nat := row10891_small ++ row10891_block_005
theorem row10891_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10891 row10891_large_left_left :=
  Budget.RowUpperBound.branch row10891_small_covers row10891_block_005_covers

def row10891_large_left_right_right : List Nat := row10891_block_007 ++ row10891_block_008
theorem row10891_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10891 row10891_large_left_right_right :=
  Budget.RowUpperBound.branch row10891_block_007_covers row10891_block_008_covers

def row10891_large_left_right : List Nat := row10891_block_006 ++ row10891_large_left_right_right
theorem row10891_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10891 row10891_large_left_right :=
  Budget.RowUpperBound.branch row10891_block_006_covers row10891_large_left_right_right_covers

def row10891_large_left : List Nat := row10891_large_left_left ++ row10891_large_left_right
theorem row10891_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10891 row10891_large_left :=
  Budget.RowUpperBound.branch row10891_large_left_left_covers row10891_large_left_right_covers

def row10891_large_right_left : List Nat := row10891_block_009 ++ row10891_block_010
theorem row10891_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10891 row10891_large_right_left :=
  Budget.RowUpperBound.branch row10891_block_009_covers row10891_block_010_covers

def row10891_large_right_right_right : List Nat := row10891_block_012 ++ row10891_block_013
theorem row10891_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10891 row10891_large_right_right_right :=
  Budget.RowUpperBound.branch row10891_block_012_covers row10891_block_013_covers

def row10891_large_right_right : List Nat := row10891_block_011 ++ row10891_large_right_right_right
theorem row10891_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10891 row10891_large_right_right :=
  Budget.RowUpperBound.branch row10891_block_011_covers row10891_large_right_right_right_covers

def row10891_large_right : List Nat := row10891_large_right_left ++ row10891_large_right_right
theorem row10891_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10891 row10891_large_right :=
  Budget.RowUpperBound.branch row10891_large_right_left_covers row10891_large_right_right_covers

def row10891_large : List Nat := row10891_large_left ++ row10891_large_right
theorem row10891_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10891 row10891_large :=
  Budget.RowUpperBound.branch row10891_large_left_covers row10891_large_right_covers

def row10891 : List Nat := row10891_large
theorem row10891_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10891 row10891 := row10891_large_covers

def row10903_small : List Nat := []
theorem row10903_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10903 row10903_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10903_block_005 : List Nat := []
theorem row10903_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10903 row10903_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10903_block_006 : List Nat := [11353]
def plan10903_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 31), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .second, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row10903_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10903 row10903_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10903_block_006
  · decide
  · rfl

def row10903_block_007 : List Nat := [12547, 12583, 13381, 14083, 14341]
def plan10903_block_007 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 19), .possible, (.firstFactor 11), .first, .first, .possible, (.secondFactor 37), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 23), .first, .first, .second, .first, .first, .second, .first, (.secondFactor 7), .second, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, (.firstFactor 23), .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 37), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 31), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 19), (.secondFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10903_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10903 row10903_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10903_block_007
  · decide
  · rfl

def row10903_block_008 : List Nat := []
def plan10903_block_008 : List Budget.Choice := [(.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 37), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10903_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10903 row10903_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10903_block_008
  · decide
  · rfl

def row10903_block_009 : List Nat := []
theorem row10903_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10903 row10903_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10903_block_010 : List Nat := []
theorem row10903_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10903 row10903_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10903_block_011 : List Nat := []
theorem row10903_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10903 row10903_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10903_block_012 : List Nat := []
theorem row10903_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10903 row10903_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10903_block_013 : List Nat := []
theorem row10903_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10903 row10903_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10903_large_left_left : List Nat := row10903_small ++ row10903_block_005
theorem row10903_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10903 row10903_large_left_left :=
  Budget.RowUpperBound.branch row10903_small_covers row10903_block_005_covers

def row10903_large_left_right_right : List Nat := row10903_block_007 ++ row10903_block_008
theorem row10903_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10903 row10903_large_left_right_right :=
  Budget.RowUpperBound.branch row10903_block_007_covers row10903_block_008_covers

def row10903_large_left_right : List Nat := row10903_block_006 ++ row10903_large_left_right_right
theorem row10903_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10903 row10903_large_left_right :=
  Budget.RowUpperBound.branch row10903_block_006_covers row10903_large_left_right_right_covers

def row10903_large_left : List Nat := row10903_large_left_left ++ row10903_large_left_right
theorem row10903_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10903 row10903_large_left :=
  Budget.RowUpperBound.branch row10903_large_left_left_covers row10903_large_left_right_covers

def row10903_large_right_left : List Nat := row10903_block_009 ++ row10903_block_010
theorem row10903_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10903 row10903_large_right_left :=
  Budget.RowUpperBound.branch row10903_block_009_covers row10903_block_010_covers

def row10903_large_right_right_right : List Nat := row10903_block_012 ++ row10903_block_013
theorem row10903_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10903 row10903_large_right_right_right :=
  Budget.RowUpperBound.branch row10903_block_012_covers row10903_block_013_covers

def row10903_large_right_right : List Nat := row10903_block_011 ++ row10903_large_right_right_right
theorem row10903_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10903 row10903_large_right_right :=
  Budget.RowUpperBound.branch row10903_block_011_covers row10903_large_right_right_right_covers

def row10903_large_right : List Nat := row10903_large_right_left ++ row10903_large_right_right
theorem row10903_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10903 row10903_large_right :=
  Budget.RowUpperBound.branch row10903_large_right_left_covers row10903_large_right_right_covers

def row10903_large : List Nat := row10903_large_left ++ row10903_large_right
theorem row10903_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10903 row10903_large :=
  Budget.RowUpperBound.branch row10903_large_left_covers row10903_large_right_covers

def row10903 : List Nat := row10903_large
theorem row10903_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10903 row10903 := row10903_large_covers

end ProjectEulerSolutions.PrimePairLarge
