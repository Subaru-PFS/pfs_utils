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

    VERSION=$CLEAN_VERSION
    PREFIX=$CLEAN_PREFIX
}

