install ()
{
    #VERSION=$(git describe)
    echo "install VERSION=$VERSION PREFIX=$PREFIX"
    CLEAN_VERSION=$VERSION
    CLEAN_PREFIX=$PREFIX
    
    if test -z "$SCONSUTILS_DIR"; then
        mv SConstruct SConstruct-disabledForIcs 2>/dev/null
    fi

    default_install
}

decl ()
{
    echo "decl VERSION=$VERSION PREFIX=$PREFIX"
    VERSION=$CLEAN_VERSION
    PREFIX=$CLEAN_PREFIX
    echo "decl fixed VERSION=$VERSION PREFIX=$PREFIX"
    
    default_decl
}

